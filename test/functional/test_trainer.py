import gymnasium
import torch

from frankenstein.algorithms.ppo.trainer import RecurrentPPOTrainer, RecurrentModel


class Test_Model(RecurrentModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc_act = torch.nn.Linear(obs_dim, act_dim)
        self.fc_val = torch.nn.Linear(obs_dim, 1)

    def forward(self, *x):
        x, hidden = x
        return {
            'action': self.fc_act(x),
            'value': self.fc_val(x),
            'hidden': hidden,
        }

    def init_hidden(self, batch_size):
        return ()

    @property
    def hidden_batch_dims(self):
        return ()


def test_ppo_trainer():
    env = gymnasium.make_vec('CartPole-v1', num_envs=2)
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
    trainer.train(max_transitions=1000)
