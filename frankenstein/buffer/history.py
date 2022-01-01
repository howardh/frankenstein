from typing import Optional, Generic, TypeVar, Mapping

import torch
from torchtyping import TensorType
from torch.utils.data.dataloader import default_collate

ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')
MiscType = TypeVar('MiscType')

def transpose_batch_seqlen(data):
    """ Transpose the batch size and sequence length dimensions (i.e. dimensions 0 and 1) """
    if isinstance(data,torch.Tensor):
        return data.transpose(1,0)
    elif isinstance(data,tuple) or isinstance(data,list):
        return tuple([
            transpose_batch_seqlen(v)
            for v in data
        ])
    elif isinstance(data,Mapping):
        return {
                k: transpose_batch_seqlen(v)
                for k,v in data.items()
        }
    else:
        raise Exception(f'Unable to handle data of type {type(data)}')

class HistoryBuffer(Generic[ObsType,ActionType,MiscType]):
    def __init__(self,
            max_len : int,
            num_envs : int = 0,
            default_action : ActionType = None,
            batch_first : bool = False,
            device : torch.device = torch.device('cpu')
        ) -> None:
        """
        Args:
            max_len (int): The number of transitions to store for each environment. The `HistoryBuffer` will keep a history of `max_len+1` observations.
            num_envs (int): Number of environments running concurrently. This can grow dynamically.
            default_action: Action to use as padding at the end of an episode.
            batch_first (bool): If set to `True`, then all returned tensors will have the shape `(batch_size, seq_len, ...)`. Otherwise, they will take the shape `(seq_len, batch_size, ...)`. Default: `False`.
        """
        self._num_envs = num_envs
        self.device = device
        self.max_len = max_len
        self.batch_first = batch_first

        # Validate input
        if max_len < 1:
            raise ValueError('`max_len` must be at least 1')

        # History is saved as a list instead of tensors to save on memory. This allows for data to be stored in formats like LazyFrames.
        self.obs_history = [[] for _ in range(num_envs)]
        self.action_history = [[] for _ in range(num_envs)]
        self.reward_history = [[] for _ in range(num_envs)]
        self.terminal_history = [[] for _ in range(num_envs)]
        self.misc_history = [[] for _ in range(num_envs)]

        self.default_action = default_action
        self.default_reward = 0
    def __getitem__(self, index):
        self._resize_if_needed(index)
        return HistoryBufferSlice(self, env_index=index)
    def _resize_if_needed(self,index):
        if index >= self._num_envs:
            if index == self._num_envs:
                self.obs_history.append([])
                self.action_history.append([])
                self.reward_history.append([])
                self.terminal_history.append([])
                self.misc_history.append([])
                self._num_envs += 1
            else:
                raise Exception('The HistoryBuffer cannot be dynamically resized by more than one element at a time. Either access the slices in order, or predefine the number of environments appropriately with `num_envs`.')
    def append_obs(self,
            obs : ObsType,
            reward : Optional[float] = None,
            terminal : bool = False,
            misc : MiscType = None,
            env_index : int = None
            ) -> None:
        # Default env_index
        if env_index is None:
            if self._num_envs <= 1:
                env_index = 0
            else:
                raise Exception('`env_index` must be specified.')
        self._resize_if_needed(env_index)
        # Handle boundary between episodes
        if reward is None:
            if len(self.obs_history[env_index]) != 0:
                # The last episode just finished, so we receive a new observation to start the episode without an action in between.
                # Add an action to pad the actions list.
                self.append_action(self.default_action, env_index)
            reward = self.default_reward

        # Make sure the observations and actions are in sync
        assert len(self.obs_history) == len(self.action_history)

        # Save data
        self.obs_history[env_index].append(obs)
        self.reward_history[env_index].append(reward)
        self.terminal_history[env_index].append(terminal)
        self.misc_history[env_index].append(misc)

        # Enforce max length
        if self.max_len is not None and len(self.obs_history[env_index]) > self.max_len+1:
            self.obs_history[env_index] = self.obs_history[env_index][1:]
            self.reward_history[env_index] = self.reward_history[env_index][1:]
            self.terminal_history[env_index] = self.terminal_history[env_index][1:]
            self.action_history[env_index] = self.action_history[env_index][1:]
            self.misc_history[env_index] = self.misc_history[env_index][1:]
    def append_action(self, action : ActionType, env_index : int = None):
        # Default env_index
        if env_index is None:
            if self._num_envs <= 1:
                env_index = 0
            else:
                raise Exception('`env_index` must be specified.')
        self._resize_if_needed(env_index)
        # Append action
        obs_history = self.obs_history[env_index]
        action_history = self.action_history[env_index]
        assert len(obs_history) == len(action_history)+1
        action_history.append(action)
    def clear(self):
        for i in range(self._num_envs):
            self[i].clear()

    @property
    def obs(self) -> TensorType['seq_len','num_envs','obs_shape']:
        output = torch.stack([
            self[i].obs for i in range(self._num_envs)
        ], dim=1-self.batch_first)
        return output
    @property
    def reward(self) -> TensorType['seq_len','num_envs',float]:
        output = torch.tensor([
                [r for r in reward_hist]
                for reward_hist in self.reward_history
        ], device=self.device)
        if not self.batch_first:
            output = transpose_batch_seqlen(output)
        return output
    @property
    def terminal(self) -> TensorType['seq_len','num_envs',bool]:
        output = torch.tensor([
                [t for t in term_hist]
                for term_hist in self.terminal_history
        ], device=self.device)
        if not self.batch_first:
            output = transpose_batch_seqlen(output)
        return output
    @property
    def action(self) -> TensorType['seq_len','num_envs','action_shape']:
        output = default_collate([self[i].action for i in range(self._num_envs)])
        if not self.batch_first:
            output = transpose_batch_seqlen(output)
        return output
    @property
    def misc(self):
        output = default_collate([
            self[i].misc for i in range(self._num_envs)
        ])
        if self.batch_first:
            return output
        return transpose_batch_seqlen(output)

class HistoryBufferSlice(Generic[ObsType,ActionType,MiscType]):
    def __init__(self, buffer, env_index) -> None:
        self.buffer = buffer
        self.env_index = env_index
    def append_obs(self,
            obs : ObsType,
            reward : Optional[float] = None,
            terminal : bool = False,
            misc : MiscType = None,
            ) -> None:
        self.buffer.append_obs(obs=obs, reward=reward, terminal=terminal, misc=misc, env_index=self.env_index)
    def append_action(self, action : ActionType):
        self.buffer.append_action(action=action, env_index=self.env_index)
    def clear(self):
        i = len(self.obs_history)-1
        self.obs_history = self.obs_history[i:]
        self.reward_history = self.reward_history[i:]
        self.terminal_history = self.terminal_history[i:]
        self.action_history = self.action_history[i:]
        self.misc_history = self.misc_history[i:]

    @property
    def obs_history(self):
        return self.buffer.obs_history[self.env_index]
    @property
    def reward_history(self):
        return self.buffer.reward_history[self.env_index]
    @property
    def terminal_history(self):
        return self.buffer.terminal_history[self.env_index]
    @property
    def action_history(self):
        return self.buffer.action_history[self.env_index]
    @property
    def misc_history(self):
        return self.buffer.misc_history[self.env_index]

    @obs_history.setter
    def obs_history(self, val):
        self.buffer.obs_history[self.env_index] = val
    @reward_history.setter
    def reward_history(self, val):
        self.buffer.reward_history[self.env_index] = val
    @terminal_history.setter
    def terminal_history(self, val):
        self.buffer.terminal_history[self.env_index] = val
    @action_history.setter
    def action_history(self, val):
        self.buffer.action_history[self.env_index] = val
    @misc_history.setter
    def misc_history(self, val):
        self.buffer.misc_history[self.env_index] = val

    @property
    def obs(self) -> TensorType['seq_len','obs_shape']:
        return torch.stack([
            o if isinstance(o,torch.Tensor) else torch.tensor(o,device=self.buffer.device)
            for o in self.obs_history
        ])
    @property
    def action(self) -> TensorType['seq_len','action_shape']:
        return default_collate(self.action_history)
    @property
    def reward(self) -> TensorType['seq_len','num_envs',float]:
        return torch.tensor(self.buffer.reward_history[self.env_index], device=self.buffer.device)
    @property
    def terminal(self) -> TensorType['seq_len','num_envs',bool]:
        return torch.tensor(self.buffer.terminal_history[self.env_index], device=self.buffer.device)
    @property
    def misc(self):
        return default_collate(self.misc_history)

class VecHistoryBuffer:
    def __init__(self) -> None:
        raise NotImplementedError() # pragma: no cover
