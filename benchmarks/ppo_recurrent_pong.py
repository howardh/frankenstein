import gymnasium
import torch
import ale_py

from frankenstein.algorithms.ppo.trainer import RecurrentPPOTrainer, RecurrentModel


if __name__ == '__main__':
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
        def forward(self, x, hidden):
            x = x.permute(0, 3, 1, 2).float()
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) # type: ignore

    trainer = RecurrentPPOTrainer(
            env = env,
            model = model,
            optimizer = optimizer,
            device = device,
    )
    trainer.train()
