import argparse

import gymnasium
import torch
import ale_py

from frankenstein.algorithms.ppo.trainer import RecurrentPPOTrainer, RecurrentModel
from frankenstein.algorithms.trainer import Checkpoint

HIDDEN_SIZE = 512
class Model(RecurrentModel):
    def __init__(self, obs_shape, act_dim):
        assert obs_shape == (210, 160, 3)
        super().__init__()
        self.conv = torch.nn.Sequential(
            # (3, 210, 160)
            torch.nn.Conv2d(3, 32, 8, stride=4),
            torch.nn.ReLU(),
            # (32, 51, 39)
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            # (64, 24, 18)
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
            # (64, 22, 16)
            torch.nn.Flatten()
        )
        self.lstm = torch.nn.LSTMCell(64*22*16, HIDDEN_SIZE)
        self.fc_policy = torch.nn.Linear(HIDDEN_SIZE, act_dim)
        self.fc_value = torch.nn.Linear(HIDDEN_SIZE, 1)
    def forward(self, *inputs):
        x, hidden = inputs
        x = x.permute(0, 3, 1, 2).float() / 255
        x = self.conv(x)
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

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rollout-length', type=int, default=128)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--norm-adv', type=bool, default=True)
    parser.add_argument('--clip-pg-ratio', type=float, default=0.1)
    parser.add_argument('--clip-vf-loss', type=float, default=None)
    parser.add_argument('--vf-loss-coeff', type=float, default=0.5)
    parser.add_argument('--entropy-loss-coeff', type=float, default=0.01)
    parser.add_argument('--target-kl', type=float, default=0.01)
    parser.add_argument('--num-epochs', type=int, default=4)
    parser.add_argument('--backtrack', action='store_true')
    parser.add_argument('--max-grad-norm', type=float, default=None)

    return parser


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device', device)

    gymnasium.register_envs(ale_py)

    env = gymnasium.make_vec('ALE/Pong-v5', num_envs=8, vectorization_mode='async')

    assert isinstance(env.single_observation_space, gymnasium.spaces.Box)
    assert isinstance(env.single_action_space, gymnasium.spaces.Discrete)

    model = Model(env.single_observation_space.shape, env.single_action_space.n)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = RecurrentPPOTrainer(
            env = env,
            model = model,
            optimizer = optimizer,
            device = device,
            config = {
                'rollout_length': args.rollout_length,
                'warmup_steps': args.warmup_steps,
                'discount': args.discount,
                'gae_lambda': args.gae_lambda,
                'norm_adv': args.norm_adv,
                'clip_pg_ratio': args.clip_pg_ratio,
                'clip_vf_loss': args.clip_vf_loss,
                'vf_loss_coeff': args.vf_loss_coeff,
                'entropy_loss_coeff': args.entropy_loss_coeff,
                'target_kl': args.target_kl,
                'num_epochs': args.num_epochs,
                'backtrack': args.backtrack,
                'max_grad_norm': args.max_grad_norm,
            }
    )
    if args.checkpoint is not None:
        checkpoint = Checkpoint({
            'model': model,
            'optimizer': optimizer,
        }, frequency=(30, 'minutes'), path=args.checkpoint)
    else:
        checkpoint = None
    trainer.train(max_transitions=5_000_000, checkpoint=checkpoint)


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)
