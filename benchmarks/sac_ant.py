import argparse
import itertools

import gymnasium
import torch

from frankenstein.algorithms.sac.trainer import FeedforwardSACTrainer
from frankenstein.algorithms.utils import FeedforwardModel
from frankenstein.algorithms.trainer import Checkpoint


LOGSTD_MIN = -5
LOGSTD_MAX = 2


class ActorModel(FeedforwardModel):
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
    def forward(self, x):
        x = x.float()
        x = self.fc(x)
        mean = self.fc_action_mean(x)
        logstd = self.fc_action_std(x)
        logstd = torch.tanh(logstd)
        logstd= 0.5 * (logstd + 1) * (LOGSTD_MAX - LOGSTD_MIN) + LOGSTD_MIN  # y = 2[(x - min) / (max - min) - 0.5] -> x = (y + 1) * (max - min) / 2 + min
        return {
            'action_mean': mean,
            'action_logstd': logstd,
        }


class CriticModel(FeedforwardModel):
    def __init__(self, obs_dim, act_dim, hidden_dims=64):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(obs_dim+act_dim, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, 1)
        )
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = x.float()
        return { 'value': self.fc(x) }


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default=None)

    return parser


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device', device)

    env = gymnasium.make_vec(
        'Ant-v4', num_envs=1,
        wrappers=[
            lambda env: gymnasium.wrappers.RecordEpisodeStatistics(env),
        ]
    )
    assert isinstance(env.single_observation_space, gymnasium.spaces.Box)
    assert isinstance(env.single_action_space, gymnasium.spaces.Box)
    actor_model = ActorModel(env.single_observation_space.shape[0], env.single_action_space.shape[0], hidden_dims=256)
    critic_model_1 = CriticModel(env.single_observation_space.shape[0], env.single_action_space.shape[0], hidden_dims=256)
    critic_model_2 = CriticModel(env.single_observation_space.shape[0], env.single_action_space.shape[0], hidden_dims=256)

    actor_model.to(device)
    critic_model_1.to(device)
    critic_model_2.to(device)
    actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(itertools.chain(critic_model_1.parameters(), critic_model_2.parameters()), lr=1e-3)

    trainer = FeedforwardSACTrainer(
            env = env,
            actor_model = actor_model,
            critic_model_1 = critic_model_1,
            critic_model_2 = critic_model_2,
            actor_optimizer = actor_optimizer,
            critic_optimizer = critic_optimizer,
            device = device,
            config={
                'buffer_size': 1_000_000,
                'warmup_steps': 5_000,
                'batch_size': 256,
                'discount': 0.99,
                'policy_update_frequency': 2,
                'target_update_frequency': 1,
                'target_update_rate': 0.005,
                'update_frequency': 1,
                'entropy_coeff': 0.05,
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
    trainer.train(max_transitions=1_000_000, checkpoint=checkpoint)


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)
