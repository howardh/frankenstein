from typing import Mapping, overload

from jaxtyping import Float, Bool, Real
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from .views import TrajectoryView, TransitionView, Trajectory, Transition


@overload
def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor: ...

@overload
def to_device(data: tuple, device: torch.device) -> tuple: ...

@overload
def to_device(data: list, device: torch.device) -> tuple: ...

@overload
def to_device(data: Mapping, device: torch.device) -> dict: ...

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple) or isinstance(data, list):
        return tuple([
            to_device(v, device)
            for v in data
        ])
    elif isinstance(data, Mapping):
        return {
            k: to_device(v, device)
            for k, v in data.items()
        }
    else:
        raise Exception(f'Unable to handle data of type {type(data)}')  # pragma: no cover


class VecHistoryBuffer():
    def __init__(self,
                 max_len: int,
                 num_envs: int,
                 trajectory_length: int | None = None,
                 device: torch.device = torch.device('cpu')
                 ) -> None:
        self._num_envs = num_envs
        self.device = device
        self.max_len = max_len
        self.trajectory_length = trajectory_length

        # Validate input
        if max_len < 1:
            raise ValueError('`max_len` must be at least 1')
        if num_envs < 1:
            raise ValueError('`num_envs` must be at least 1')

        self.obs_history = []
        self.action_history = []
        self.reward_history = []
        self.terminated_history = []
        self.truncated_history = []
        self.misc_history = []

        self.default_reward = np.zeros(num_envs, dtype=float)

        self._num_transitions = np.zeros(num_envs, dtype=int)
        self._transition_view = TransitionView(self)
        self._trajectory_view = TrajectoryView(self)

    def append_obs(self,
                   obs,
                   reward: np.ndarray | None = None,
                   terminated: np.ndarray | None = None,
                   truncated: np.ndarray | None = None,
                   misc = None,
                   ) -> None:
        # Handle boundary between episodes
        if reward is None:
            reward = self.default_reward
        if terminated is None:
            terminated = np.zeros(self._num_envs, dtype=bool)
        if truncated is None:
            truncated = np.zeros(self._num_envs, dtype=bool)
        if misc is None:
            misc = [None] * self._num_envs

        # Validate inputs
        if __debug__:
            if isinstance(obs, dict):
                for o in obs.values():
                    assert len(o) == self._num_envs
            else:
                assert len(obs) == self._num_envs
            assert len(reward) == self._num_envs
            assert len(terminated) == self._num_envs
            assert len(truncated) == self._num_envs

        # Increment transition count
        if len(self.terminated_history) > 0:
            done = self.terminated_history[-1] | self.truncated_history[-1]
            self._num_transitions += 1 - done

        # Make sure the observations and actions are in sync
        assert len(self.obs_history) == len(self.action_history)

        # Save data
        self.obs_history.append(obs)
        self.reward_history.append(reward)
        self.terminated_history.append(terminated)
        self.truncated_history.append(truncated)
        self.misc_history.append(misc)

        # Enforce max length
        if self.max_len is not None and len(self.obs_history) > self.max_len:
            self._num_transitions -= ~(self.terminated_history[0] | self.truncated_history[0])
            self.obs_history = self.obs_history[1:]
            self.reward_history = self.reward_history[1:]
            self.terminated_history = self.terminated_history[1:]
            self.truncated_history = self.truncated_history[1:]
            self.action_history = self.action_history[1:]
            self.misc_history = self.misc_history[1:]

    def append_action(self, action):
        # Validation
        assert len(action) == self._num_envs
        # Append action
        obs_history = self.obs_history
        action_history = self.action_history
        assert len(obs_history) == len(action_history)+1
        action_history.append(action)

    def clear(self, fullclear: bool = False) -> None:
        """
        Clear the history buffer.

        Args:
            fullclear: If `True`, clear everything. Otherwise, keep the last element in the buffer. defaults to `False`.
        """
        if fullclear:
            self.obs_history = []
            self.action_history = []
            self.reward_history = []
            self.terminated_history = []
            self.truncated_history = []
            self.misc_history = []
        else:
            i = len(self.obs_history)-1
            self.obs_history = self.obs_history[i:]
            self.reward_history = self.reward_history[i:]
            self.terminated_history = self.terminated_history[i:]
            self.truncated_history = self.truncated_history[i:]
            self.action_history = self.action_history[i:]
            self.misc_history = self.misc_history[i:]

    @property
    def num_transitions(self):
        return self._num_transitions.sum()

    def get_transition(self, index) -> Transition:
        if index >= self.num_transitions:
            raise IndexError('Index out of range')

        # Determine which envionment we're selecting from, and the index of the transition within that environment
        env_index = 0
        idx_in_env = index
        for n in self._num_transitions:
            if idx_in_env >= n:
                idx_in_env -= n
                env_index += 1

        # Adjust the transition index for terminations
        i = 0
        for i,(term,trunc) in enumerate(zip(self.terminated_history, self.truncated_history)):
            done = term[env_index] or trunc[env_index]
            if done:
                idx_in_env += 1
            if i == idx_in_env:
                break

        #term = np.array(self.terminated_history)
        #trun = np.array(self.truncated_history)
        #done = term | trun
        #x = np.cumsum(done)

        #env_index = 0
        #i = 0

        return Transition(
            torch.tensor(self.obs_history[i][env_index], device=self.device),
            torch.tensor(self.action_history[i][env_index], device=self.device),
            torch.tensor(self.obs_history[i+1][env_index], device=self.device),
            self.reward_history[i+1][env_index],
            self.terminated_history[i+1][env_index],
            self.truncated_history[i+1][env_index],
            self.misc_history[i][env_index],
            self.misc_history[i+1][env_index],
        )

    def get_random_transition(self) -> Transition:
        while True:
            env_index = np.random.randint(0, self._num_envs)
            i = np.random.randint(0, len(self.obs_history)-1)
            
            # Check if transition is valid
            if self.terminated_history[i][env_index] or self.truncated_history[i][env_index]:
                continue
            
            break

        return Transition(
            torch.tensor(self.obs_history[i][env_index], device=self.device),
            torch.tensor(self.action_history[i][env_index], device=self.device),
            torch.tensor(self.obs_history[i+1][env_index], device=self.device),
            self.reward_history[i+1][env_index],
            self.terminated_history[i+1][env_index],
            self.truncated_history[i+1][env_index],
            self.misc_history[i][env_index],
            self.misc_history[i+1][env_index],
        )

    @property
    def num_trajectories(self):
        if self.trajectory_length is None:
            raise ValueError('Trajectory sampling is disabled. To enable, set `trajectory_length` to a positive integer.')
        num_obs_per_env = len(self.obs_history)
        if num_obs_per_env < self.trajectory_length:
            return 0
        return (num_obs_per_env - self.trajectory_length) * self._num_envs

    def get_trajectory(self, index) -> Trajectory:
        if index >= self.num_trajectories:
            raise IndexError('Index out of range')
        assert self.trajectory_length is not None

        num_obs_per_env = len(self.obs_history)
        trajectories_per_env = num_obs_per_env - self.trajectory_length
        env_index = index // trajectories_per_env
        step_index = index % trajectories_per_env

        return Trajectory(
            obs = default_collate([o[env_index] for o in self.obs_history[step_index:step_index+self.trajectory_length]]).to(self.device),
            action = default_collate([a[env_index] for a in self.action_history[step_index:step_index+self.trajectory_length]]).to(self.device),
            next_obs = default_collate([o[env_index] for o in self.obs_history[step_index+1:step_index+self.trajectory_length+1]]).to(self.device),
            reward = torch.tensor([r[env_index] for r in self.reward_history[step_index+1:step_index+self.trajectory_length+1]]).to(self.device),
            terminated = torch.tensor([t[env_index] for t in self.terminated_history[step_index+1:step_index+self.trajectory_length+1]]).to(self.device),
            truncated = torch.tensor([t[env_index] for t in self.truncated_history[step_index+1:step_index+self.trajectory_length+1]]).to(self.device),
            misc = [m[env_index] for m in self.misc_history[step_index:step_index+self.trajectory_length]],
            next_misc = [m[env_index] for m in self.misc_history[step_index+1:step_index+self.trajectory_length+1]]
        )

    @property
    def transitions(self):
        return self._transition_view

    @property
    def trajectories(self):
        return self._trajectory_view


    @property
    def obs(self) -> Real[torch.Tensor, 'seq_len num_envs obs_shape']:
        output = default_collate(self.obs_history)
        output = to_device(output, self.device)
        return output

    @property
    def reward(self) -> Float[torch.Tensor, 'seq_len num_envs']:
        output = torch.stack([
            torch.tensor(x, device=self.device) for x in self.reward_history
        ], dim=0)
        return output

    @property
    def terminated(self) -> Bool[torch.Tensor, 'seq_len num_envs']:
        output = torch.stack([
            torch.tensor(x, device=self.device) for x in self.terminated_history
        ], dim=0)
        return output

    @property
    def action(self) -> Real[torch.Tensor, 'seq_len num_envs action_shape']:
        if len(self.action_history) == 0:
            return torch.zeros(0, self._num_envs, 0, device=self.device)
        output = default_collate(self.action_history)
        output = to_device(output, self.device)
        return output

    @property
    def misc(self):
        elem = self.misc_history[0]
        if elem is None:
            return None
        if isinstance(elem, Mapping):
            output = default_collate(self.misc_history)
        else:
            output = default_collate([default_collate(x) for x in self.misc_history])
        output = to_device(output, self.device)
        return output
