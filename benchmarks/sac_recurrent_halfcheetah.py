import argparse
import itertools

import gymnasium
from gymnasium.wrappers.vector import RecordEpisodeStatistics, TransformObservation
import torch

from frankenstein.algorithms.sac.trainer import RecurrentSACTrainer
from frankenstein.algorithms.utils import RecurrentModel
from frankenstein.algorithms.trainer import Checkpoint


LOGSTD_MIN = -5
LOGSTD_MAX = 2


class ActorModel(RecurrentModel):
    def __init__(self, obs_dim, act_dim, hidden_dims=64):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.fc = torch.nn.Linear(obs_dim, hidden_dims)
        self.lstm = torch.nn.LSTM(hidden_dims, hidden_dims)
        self.fc_action_mean = torch.nn.Linear(hidden_dims, act_dim)
        self.fc_action_std = torch.nn.Linear(hidden_dims, act_dim)
        self.lstm.flatten_parameters()
    def forward(self, *inputs):
        return self.forward_step(inputs[0], hidden=inputs[1])
    def forward_sequence(self, *inputs, hidden):
        x, = inputs
        x = x.float()
        x = self.fc(x)
        x = torch.relu(x)
        # hidden is provided with shape [batch_size, hidden_dims]
        # Reshape to (sequence_length, batch_size, hidden_dims) for lstm.
        x, (h,c) = self.lstm(x, (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)))
        # Now h, c have shape (1, hidden_batch_size, hidden_dims)
        # Bring them back to (batch_size, hidden_dims)
        # x has shape (sequence_length, batch_size, hidden_dims), which is correct
        h = h.squeeze(0)
        c = c.squeeze(0)
        mean = self.fc_action_mean(x)
        logstd = self.fc_action_std(x)
        logstd = torch.tanh(logstd)
        logstd= 0.5 * (logstd + 1) * (LOGSTD_MAX - LOGSTD_MIN) + LOGSTD_MIN  # y = 2[(x - min) / (max - min) - 0.5] -> x = (y + 1) * (max - min) / 2 + min
        return {
            'action_mean': mean,
            'action_logstd': logstd,
            'hidden': (h, c),
        }
    def forward_step(self, *inputs, hidden):
        x, = inputs
        x = x.float()
        x = self.fc(x)
        x = torch.relu(x)
        # Expected x shape: (batch_size, hidden_dims)
        # We need it in the shape (sequence_length=1, batch_size, hidden_dims) for lstm.
        x, (h,c) = self.lstm(x.unsqueeze(0), (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)))
        # Now x, h, c have shape (1, hidden_batch_size, hidden_dims)
        # Bring them back to (batch_size, hidden_dims)
        x = x.squeeze(0)
        h = h.squeeze(0)
        c = c.squeeze(0)
        mean = self.fc_action_mean(x)
        logstd = self.fc_action_std(x)
        logstd = torch.tanh(logstd)
        logstd= 0.5 * (logstd + 1) * (LOGSTD_MAX - LOGSTD_MIN) + LOGSTD_MIN  # y = 2[(x - min) / (max - min) - 0.5] -> x = (y + 1) * (max - min) / 2 + min
        return {
            'action_mean': mean,
            'action_logstd': logstd,
            'hidden': (h, c),
        }
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_dims, device=device),
            torch.zeros(batch_size, self.hidden_dims, device=device),
        )
    @property
    def hidden_batch_dims(self):
        return (0, 0)


class CriticModel(RecurrentModel):
    def __init__(self, obs_dim, act_dim, hidden_dims=64):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.fc_in = torch.nn.Linear(obs_dim+act_dim, hidden_dims)
        self.lstm = torch.nn.LSTM(hidden_dims, hidden_dims)
        self.fc_out = torch.nn.Linear(hidden_dims, 1)
        self.lstm.flatten_parameters()
    def forward(self, *inputs):
        return self.forward_step(inputs[0], inputs[1], hidden=inputs[2])
    def forward_sequence(self, *inputs, hidden):
        x, a = inputs
        x = torch.cat([x, a], dim=2)  # Concatenate along feature dimension
        x = x.float()
        x = self.fc_in(x)
        # hidden is provided with shape [batch_size, hidden_dims]
        # Reshape to (sequence_length, batch_size, hidden_dims) for lstm.
        x, (h,c) = self.lstm(x, (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)))
        # Now h, c have shape (1, hidden_batch_size, hidden_dims)
        # Bring them back to (batch_size, hidden_dims)
        # x has shape (sequence_length, batch_size, hidden_dims), which is correct
        h = h.squeeze(0)
        c = c.squeeze(0)
        x = self.fc_out(x)
        # x has shape (sequence_length, batch_size, 1)
        # Get rid of the last dimension
        x = x.squeeze(-1)
        return {
            'value': x,
            'hidden': (h, c),
        }
    def forward_step(self, *inputs, hidden):
        x, a = inputs
        x = torch.cat([x, a], dim=1)
        x = x.float()
        x = self.fc_in(x)
        # Expected x shape: (batch_size, hidden_dims)
        # We need it in the shape (sequence_length=1, batch_size, hidden_dims) for lstm.
        x, (h,c) = self.lstm(x.unsqueeze(0), (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)))
        # Now x, h, c have shape (1, hidden_batch_size, hidden_dims)
        # Bring them back to (batch_size, hidden_dims)
        x = x.squeeze(0)
        h = h.squeeze(0)
        c = c.squeeze(0)
        return {
            'value': self.fc_out(x),
            'hidden': (h, c),
        }
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_dims, device=device),
            torch.zeros(batch_size, self.hidden_dims, device=device),
        )
    @property
    def hidden_batch_dims(self):
        return (0, 0)


def init_arg_parser():
    parser = argparse.ArgumentParser()

    # Algorithm parameters
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--trajectory-length', type=int, default=128)
    parser.add_argument('--warmup-steps', type=int, default=5_000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--policy-update-interval', type=int, default=2,
                        help='Number of transitions between updates to the policy network. Note that this does not change the total number of gradient steps applied to the policy network. An update interval of N means that N policy gradient steps are taken after ever N transitions.')
    parser.add_argument('--critic-target-update-interval', type=int, default=1)
    parser.add_argument('--critic-target-update-rate', type=float, default=0.005)
    parser.add_argument('--entropy-coefficient', type=float, default=0.05)

    # Model parameters
    parser.add_argument('--actor-hidden-size', type=int, default=64)
    parser.add_argument('--critic-hidden-size', type=int, default=64)

    # ...
    parser.add_argument('--max-transitions', type=int, default=10_000_000)
    parser.add_argument('--checkpoint', type=str, default=None)

    return parser


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device', device)

    env = gymnasium.make_vec(
        'HalfCheetah-v5', num_envs=2,
    )
    env = RecordEpisodeStatistics(env)
    ospace = env.single_observation_space
    env = TransformObservation(env, lambda obs: obs[:,:8], single_observation_space=gymnasium.spaces.Box(ospace.low[:8], ospace.high[:8], (8,), ospace.dtype)) # type: ignore
    assert isinstance(env.single_observation_space, gymnasium.spaces.Box)
    assert isinstance(env.single_action_space, gymnasium.spaces.Box)

    actor_model = ActorModel(env.single_observation_space.shape[0], env.single_action_space.shape[0], hidden_dims=args.actor_hidden_size)
    critic_model_1 = CriticModel(env.single_observation_space.shape[0], env.single_action_space.shape[0], hidden_dims=args.critic_hidden_size)
    critic_model_2 = CriticModel(env.single_observation_space.shape[0], env.single_action_space.shape[0], hidden_dims=args.critic_hidden_size)

    actor_model.to(device)
    critic_model_1.to(device)
    critic_model_2.to(device)
    actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(itertools.chain(critic_model_1.parameters(), critic_model_2.parameters()), lr=args.critic_lr)

    trainer = RecurrentSACTrainer(
            env = env,
            actor_model = actor_model,
            critic_model_1 = critic_model_1,
            critic_model_2 = critic_model_2,
            actor_optimizer = actor_optimizer,
            critic_optimizer = critic_optimizer,
            device = device,
            config={
                'buffer_size': args.buffer_size,
                'trajectory_length': args.trajectory_length,
                'warmup_steps': args.warmup_steps,
                'batch_size': args.batch_size,
                'discount': args.discount,
                'policy_update_interval': args.policy_update_interval,
                'target_update_interval': args.critic_target_update_interval,
                'target_update_rate': args.critic_target_update_rate,
                'entropy_coeff': args.entropy_coefficient,
            }
    )
    if args.checkpoint is not None:
        checkpoint = Checkpoint({
            'trainer': trainer,
        }, frequency=(30, 'minutes'), path=args.checkpoint)
    else:
        checkpoint = None
    trainer.train(max_transitions=args.max_transitions, checkpoint=checkpoint)


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)
