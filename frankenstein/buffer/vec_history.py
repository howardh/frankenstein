from typing import Optional, Generic, TypeVar

import numpy as np
import torch
from torchtyping import TensorType
from torch.utils.data.dataloader import default_collate

from .history import to_device

ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')
MiscType = TypeVar('MiscType')


class VecHistoryBuffer(Generic[ObsType, ActionType, MiscType]):
    def __init__(self,
                 max_len: int,
                 num_envs: int = 0,
                 device: torch.device = torch.device('cpu')
                 ) -> None:
        self._num_envs = num_envs
        self.device = device
        self.max_len = max_len

        # Validate input
        if max_len < 1:
            raise ValueError('`max_len` must be at least 1')

        self.obs_history = []
        self.action_history = []
        self.reward_history = []
        self.terminal_history = []
        self.misc_history = []

        self.default_reward = np.array([0]*num_envs)

    def append_obs(self,
                   obs: ObsType,
                   reward: Optional[np.ndarray] = None,
                   terminal: np.ndarray = None,
                   misc: MiscType = None,
                   ) -> None:
        # Handle boundary between episodes
        if reward is None:
            reward = self.default_reward
        if terminal is None:
            terminal = np.array([False]*self._num_envs)

        # Make sure the observations and actions are in sync
        assert len(self.obs_history) == len(self.action_history)

        # Save data
        self.obs_history.append(obs)
        self.reward_history.append(reward)
        self.terminal_history.append(terminal)
        self.misc_history.append(misc)

        # Enforce max length
        if self.max_len is not None and len(self.obs_history) > self.max_len+1:
            self.obs_history = self.obs_history[1:]
            self.reward_history = self.reward_history[1:]
            self.terminal_history = self.terminal_history[1:]
            self.action_history = self.action_history[1:]
            self.misc_history = self.misc_history[1:]

    def append_action(self, action: ActionType):
        # Append action
        obs_history = self.obs_history
        action_history = self.action_history
        assert len(obs_history) == len(action_history)+1
        action_history.append(action)

    def clear(self):
        i = len(self.obs_history)-1
        self.obs_history = self.obs_history[i:]
        self.reward_history = self.reward_history[i:]
        self.terminal_history = self.terminal_history[i:]
        self.action_history = self.action_history[i:]
        self.misc_history = self.misc_history[i:]

    @property
    def obs(self) -> TensorType['seq_len', 'num_envs', 'obs_shape']:
        output = torch.stack([
            torch.tensor(x, device=self.device) for x in self.obs_history
        ], dim=0,)
        return output

    @property
    def reward(self) -> TensorType['seq_len', 'num_envs', float]:
        output = torch.stack([
            torch.tensor(x, device=self.device) for x in self.reward_history
        ], dim=0)
        return output

    @property
    def terminal(self) -> TensorType['seq_len', 'num_envs', bool]:
        output = torch.stack([
            torch.tensor(x, device=self.device) for x in self.terminal_history
        ], dim=0)
        return output

    @property
    def action(self) -> TensorType['seq_len', 'num_envs', 'action_shape']:
        if len(self.action_history) == 0:
            return torch.zeros(0, self._num_envs, 0, device=self.device)
        return torch.stack([
            torch.tensor(x, device=self.device) for x in self.action_history
        ], dim=0)

    @property
    def misc(self):
        elem = self.misc_history[0]
        if isinstance(elem, dict):
            output = default_collate(self.misc_history)
        else:
            output = default_collate([default_collate(x) for x in self.misc_history])
        output = to_device(output, self.device)
        return output
