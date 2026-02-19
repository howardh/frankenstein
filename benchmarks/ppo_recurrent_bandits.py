import argparse

import gymnasium
import torch

from frankenstein.algorithms.ppo.trainer import RecurrentPPOTrainer, RecurrentModel
import frankenstein_lab.environments as fkn_envs
from frankenstein.algorithms.trainer import Checkpoint


HIDDEN_SIZE = 64
class Model(RecurrentModel):
    def __init__(self, num_arms):
        super().__init__()
        self._num_arms = num_arms
        input_size = num_arms + 2
        self.lstm = torch.nn.LSTMCell(input_size, HIDDEN_SIZE)
        self.fc_policy = torch.nn.Linear(HIDDEN_SIZE, num_arms)
        self.fc_value = torch.nn.Linear(HIDDEN_SIZE, 1)
    def forward(self, *inputs):
        x, hidden = inputs
        action = torch.nn.functional.one_hot(
            x['action'].long(),
            num_classes=self._num_arms
        )
        x = torch.cat([action, x['reward'].unsqueeze(-1), x['time'].unsqueeze(-1)/100], dim=-1)
        h, c = self.lstm(x, hidden)
        return {
            'action': self.fc_policy(h),
            'value': self.fc_value(h),
            'hidden': (h, c),
        }
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, HIDDEN_SIZE, device=device),
            torch.zeros(batch_size, HIDDEN_SIZE, device=device),
        )
    @property
    def hidden_batch_dims(self):
        return (0, 0)


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

    gymnasium.register_envs(fkn_envs)

    env = gymnasium.make_vec(
            'frankenstein/CategoricalBandits-v0',
            num_envs=8,
            vectorization_mode='async',
            reward_probabilities=[[1., 0.], [0., 1.]],
            wrappers = [
                lambda env: gymnasium.wrappers.TimeLimit(env, max_episode_steps=100), # type: ignore
            ]
    )
    assert isinstance(env.single_action_space, gymnasium.spaces.Discrete)
    model = Model(env.single_action_space.n)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # type: ignore

    trainer = RecurrentPPOTrainer(
            env = env,
            model = model,
            optimizer = optimizer,
            device = device,
            config={
                'rollout_length': 100,
                'gae_lambda': 0.3,
                #'gae_lambda': 0,
                'backtrack': True,
                'entropy_loss_coeff': 0.1,
                'target_kl': 0.05,
                'num_epochs': 6,
            }
    )
    if args.checkpoint is not None:
        checkpoint = Checkpoint({
            'model': model,
            'optimizer': optimizer,
        }, frequency=(30, 'minutes'), path=args.checkpoint)
    else:
        checkpoint = None
    trainer.train(max_transitions=1_000_000, checkpoint=checkpoint)


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)
