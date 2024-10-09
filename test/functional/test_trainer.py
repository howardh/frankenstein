import gymnasium
import torch

from frankenstein.algorithms.ppo.trainer import RecurrentPPOTrainer, RecurrentModel


class Test_Model(RecurrentModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = torch.nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        return {'action': self.fc(x)}


def test_ppo_trainer():
    env = gymnasium.vector.make('CartPole-v1', num_envs=2)
    assert env.single_observation_space.shape is not None
    assert isinstance(env.single_action_space, gymnasium.spaces.Discrete)
    model = Test_Model(env.single_observation_space.shape[0], env.single_action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = RecurrentPPOTrainer(
            env = env,
            model = model,
            optimizer = optimizer,
            device = torch.device('cpu'),
    )
    trainer.train()

    assert False
