import gymnasium
import torch

from frankenstein.algorithms.ppo.trainer import FeedforwardPPOTrainer, FeedforwardModel


if __name__ == '__main__':
    class Model(FeedforwardModel):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.values = torch.nn.Parameter(
                    torch.zeros([obs_dim])
            )
            self.policy = torch.nn.Parameter(
                    torch.zeros([obs_dim, act_dim])
            )
        def forward(self, x):
            x = x.long()
            return {
                'action': self.policy[x,:],
                'value': self.values[x].unsqueeze(1),
            }

    env = gymnasium.make_vec('FrozenLake-v1', num_envs=4)
    assert isinstance(env.single_observation_space, gymnasium.spaces.Discrete)
    assert isinstance(env.single_action_space, gymnasium.spaces.Discrete)
    model = Model(env.single_observation_space.n, env.single_action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) # type: ignore

    trainer = FeedforwardPPOTrainer(
            env = env,
            model = model,
            optimizer = optimizer,
            device = torch.device('cpu'),
    )
    trainer.train()
