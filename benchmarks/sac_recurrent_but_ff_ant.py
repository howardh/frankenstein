"""
Ant with the recurrent SAC implementation, but parameters selected to mimic the feedforward version as closely as possible. This serves as a sanity check.
"""

import argparse
import itertools

import gymnasium
import torch

from frankenstein.algorithms.sac.trainer import RecurrentSACTrainer
from frankenstein.algorithms.utils import RecurrentModel
from frankenstein.algorithms.trainer import Checkpoint


LOGSTD_MIN = -5
LOGSTD_MAX = 2


class ActorModel(RecurrentModel):
    def __init__(self, obs_dim, act_dim, hidden_dims=64):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, hidden_dims),
            torch.nn.ReLU(),
        )
        self.fc_action_mean = torch.nn.Linear(hidden_dims, act_dim)
        self.fc_action_std = torch.nn.Linear(hidden_dims, act_dim)
    def forward(self, *inputs):
        x, hidden = inputs
        x = x.float()
        x = self.fc(x)
        mean = self.fc_action_mean(x)
        logstd = self.fc_action_std(x)
        logstd = torch.tanh(logstd)
        logstd= 0.5 * (logstd + 1) * (LOGSTD_MAX - LOGSTD_MIN) + LOGSTD_MIN  # y = 2[(x - min) / (max - min) - 0.5] -> x = (y + 1) * (max - min) / 2 + min
        return {
            'action_mean': mean,
            'action_logstd': logstd,
            'hidden': hidden,
        }
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, 0, device=device),
            torch.zeros(batch_size, 0, device=device),
        )
    @property
    def hidden_batch_dims(self):
        return (0, 0)


class CriticModel(RecurrentModel):
    def __init__(self, obs_dim, act_dim, hidden_dims=64):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(obs_dim+act_dim, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, 1)
        )
    def forward(self, *inputs):
        x, a, hidden = inputs
        x = torch.cat([x, a], dim=1)
        x = x.float()
        return {
            'value': self.fc(x),
            'hidden': hidden,
        }
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, 0, device=device),
            torch.zeros(batch_size, 0, device=device),
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
    parser.add_argument('--warmup-steps', type=int, default=5_000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--policy-update-interval', type=int, default=2,
                        help='Number of transitions between updates to the policy network. Note that this does not change the total number of gradient steps applied to the policy network. An update interval of N means that N policy gradient steps are taken after ever N transitions.')
    parser.add_argument('--critic-target-update-interval', type=int, default=1)
    parser.add_argument('--critic-target-update-rate', type=float, default=0.005)
    parser.add_argument('--entropy-coefficient', type=float, default=0.05)

    # Model parameters
    parser.add_argument('--actor-hidden-size', type=int, default=256)
    parser.add_argument('--critic-hidden-size', type=int, default=256)

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
        'Ant-v4', num_envs=2,
        wrappers=[
            lambda env: gymnasium.wrappers.RecordEpisodeStatistics(env),
        ]
    )
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
                'trajectory_length': 1,
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
            'actor_model': actor_model,
            'critic_model_1': critic_model_1,
            'critic_model_2': critic_model_2,
            'actor_optimizer': actor_optimizer,
            'critic_optimizer': critic_optimizer,
            'critic_model_target_1': trainer.critic_model_target_1,
            'critic_model_target_2': trainer.critic_model_target_2,
        }, frequency=(30, 'minutes'), path=args.checkpoint)
    else:
        checkpoint = None
    trainer.train(max_transitions=args.max_transitions, checkpoint=checkpoint)


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)
