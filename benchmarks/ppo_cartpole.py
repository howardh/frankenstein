import gymnasium
import torch

from frankenstein.algorithms.ppo.trainer import FeedforwardPPOTrainer, FeedforwardModel


if __name__ == '__main__':
    class Model(FeedforwardModel):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            hidden_dims = 64
            self.fc = torch.nn.Linear(obs_dim, hidden_dims)
            self.fc_action = torch.nn.Linear(hidden_dims, act_dim)
            self.fc_value= torch.nn.Linear(hidden_dims, 1)
        def forward(self, *inputs):
            x, = inputs
            x = torch.relu(self.fc(x))
            return {
                'action': self.fc_action(x),
                'value': self.fc_value(x)
            }

    env = gymnasium.make_vec('CartPole-v1', num_envs=4)
    assert env.single_observation_space.shape is not None
    assert isinstance(env.single_action_space, gymnasium.spaces.Discrete)
    model = Model(env.single_observation_space.shape[0], env.single_action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) # type: ignore

    trainer = FeedforwardPPOTrainer(
            env = env,
            model = model,
            optimizer = optimizer,
            device = torch.device('cpu'),
    )
    trainer.train()
