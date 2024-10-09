import gymnasium
import numpy as np
import torch

from frankenstein.algorithms.ppo.trainer import FeedforwardPPOTrainer, FeedforwardModel


if __name__ == '__main__':
    class Model(FeedforwardModel):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            hidden_dims = 64
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, hidden_dims),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dims, hidden_dims),
                torch.nn.ReLU(),
            )
            self.fc_value= torch.nn.Linear(hidden_dims, 1)
            self.fc_action_mean = torch.nn.Linear(hidden_dims, act_dim)
            self.fc_action_std = torch.nn.Parameter(torch.zeros(act_dim))
        def forward(self, x):
            batch_size = x.shape[0]
            x = torch.relu(self.fc(x))
            return {
                'action_mean': self.fc_action_mean(x),
                'action_logstd': self.fc_action_std.expand(batch_size, -1),
                'value': self.fc_value(x)
            }

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device', device)

    env = gymnasium.vector.make(
        'Ant-v4', num_envs=1,
        wrappers=[
            lambda env: gymnasium.wrappers.RecordEpisodeStatistics(env), # type: ignore
            lambda env: gymnasium.wrappers.ClipAction(env), # type: ignore
            lambda env: gymnasium.wrappers.NormalizeObservation(env), # type: ignore
            lambda env: gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10)), # type: ignore
            lambda env: gymnasium.wrappers.NormalizeReward(env, gamma=0.99), # type: ignore
            lambda env: gymnasium.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10)), # type: ignore
        ]
    )
    assert isinstance(env.single_observation_space, gymnasium.spaces.Box)
    assert isinstance(env.single_action_space, gymnasium.spaces.Box)
    model = Model(env.single_observation_space.shape[0], env.single_action_space.shape[0])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # type: ignore

    trainer = FeedforwardPPOTrainer(
            env = env,
            model = model,
            optimizer = optimizer,
            device = device,
            config={
                'discount': 0.99,
                'rollout_length': 2048,
                'gae_lambda': 0.95,
                'entropy_loss_coeff': 0.0,
                'num_epochs': 10,
                'clip_pg_ratio': 0.2,
            }
    )
    trainer.train()
