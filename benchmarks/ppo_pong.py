import gymnasium
import torch
import ale_py

from frankenstein.algorithms.ppo.trainer import FeedforwardPPOTrainer, FeedforwardModel


if __name__ == '__main__':
    class Model(FeedforwardModel):
        def __init__(self, obs_shape, act_dim):
            assert obs_shape == (4, 84, 84)
            super().__init__()
            self.conv = torch.nn.Sequential(
                # (4, 84, 84)
                torch.nn.Conv2d(4, 32, 8, stride=4),
                torch.nn.ReLU(),
                # (32, 20, 20)
                torch.nn.Conv2d(32, 64, 4, stride=2),
                torch.nn.ReLU(),
                # (64, 9, 9)
                torch.nn.Conv2d(64, 64, 3, stride=1),
                torch.nn.ReLU(),
                # (64, 7, 7)
                torch.nn.Flatten(),
                torch.nn.Linear(64*7*7, 512),
            )
            self.fc_policy = torch.nn.Linear(512, act_dim)
            self.fc_value = torch.nn.Linear(512, 1)
        def forward(self, x):
            x = x / 255.0
            x = self.conv(x)
            return {
                'action': self.fc_policy(x),
                'value': self.fc_value(x),
            }

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device', device)

    gymnasium.register_envs(ale_py)

    env = gymnasium.make_vec(
            'ALE/Pong-v5',
            frameskip=1,
            num_envs=8,
            vectorization_mode='async',
            wrappers=[
                lambda env: gymnasium.wrappers.AtariPreprocessing(env, frame_skip=4), # type: ignore
                lambda env: gymnasium.wrappers.FrameStack(env, num_stack=4), # type: ignore
            ]
    )
    assert isinstance(env.single_observation_space, gymnasium.spaces.Box)
    assert isinstance(env.single_action_space, gymnasium.spaces.Discrete)
    model = Model(env.single_observation_space.shape, env.single_action_space.n)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4) # type: ignore

    trainer = FeedforwardPPOTrainer(
            env = env,
            model = model,
            optimizer = optimizer,
            device = device,
            config={
                'discount': 0.99,
                'rollout_length': 128,
                'gae_lambda': 0.95,
                'entropy_loss_coeff': 0.01,
                'num_epochs': 4,
                'clip_pg_ratio': 0.1,
                'target_kl': None,
                'backtrack': False,
                'max_grad_norm': 0.5,
            }
    )
    trainer.train()
