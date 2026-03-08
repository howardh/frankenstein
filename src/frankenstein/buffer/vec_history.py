from typing import Callable, Mapping, NamedTuple, overload, Generic, TypeVar

from jaxtyping import Float, Bool, Real
import numpy as np
from numpy.random.mtrand import f
import torch
from torch.utils.data.dataloader import default_collate

from .views import NumpyBackedTrajectoryView, TrajectoryView, TransitionView, Trajectory, Transition


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


class ListBackedVecHistoryBuffer():
    __slots__ = (
            '_num_envs', 'device', 'max_len', 'trajectory_length', 'obs_history', 'action_history', 'reward_history', 'terminated_history', 'truncated_history', 'misc_history', 'default_reward', '_num_transitions', '_transition_view', '_trajectory_view'
    )
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
        if self.trajectory_length is None:
            raise ValueError('Trajectory sampling is disabled. To enable, set `trajectory_length` to a positive integer.')

        num_obs_per_env = len(self.obs_history)
        trajectories_per_env = num_obs_per_env - self.trajectory_length
        env_index = index // trajectories_per_env
        step_index = index % trajectories_per_env

        e = env_index
        s0 = slice(step_index, step_index+self.trajectory_length)
        s1 = slice(step_index+1, step_index+self.trajectory_length+1)

        return Trajectory(
            obs = default_collate([o[e] for o in self.obs_history[s0]]).to(self.device),
            action = default_collate([a[e] for a in self.action_history[s0]]).to(self.device),
            next_obs = default_collate([o[e] for o in self.obs_history[s1]]).to(self.device),
            reward = torch.tensor([r[e] for r in self.reward_history[s1]]).to(self.device),
            terminated = torch.tensor([t[e] for t in self.terminated_history[s1]]).to(self.device),
            truncated = torch.tensor([t[e] for t in self.truncated_history[s1]]).to(self.device),
            misc = [m[e] for m in self.misc_history[s0]],
            next_misc = [m[e] for m in self.misc_history[s1]],
        )

    def get_random_trajectory(self):
        index = np.random.randint(0, self.num_trajectories)
        return self.get_trajectory(index)

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

T = TypeVar('T')
class Serializable(NamedTuple, Generic[T]):
    obs: T
    reward: T
    misc: T
    action: T

class SerializeFn(Serializable[Callable]):
    ...

default_serialize_pt_fn = lambda x: x.view(x.shape[0], -1).numpy().view(np.uint8)
default_serialize_np_fn = lambda x: x.reshape(x.shape[0], -1).view(np.uint8)
serialize_fn = SerializeFn(
    obs=default_serialize_np_fn,
    reward=default_serialize_np_fn,
    misc=lambda _: np.empty([1,0], dtype=np.uint8),
    action=default_serialize_np_fn,
)

def make_default_serde() -> tuple[SerializeFn, SerializeFn, SerializeFn]:
    """
    Create a default set of serialization and deserialization functions.

    This doesn't make any assumptions about the shape and type of the data, so it requires that the serialization function be called at least once to infer the required information.
    """
    def make(name: str):
        shape = None
        dtype = None
        def serialize(x):
            # x: [num_envs, data_shape...]
            nonlocal shape, dtype
            if shape is None or dtype is None:
                shape = x.shape
                dtype = x.dtype
            if isinstance(x, torch.Tensor):
                return x.view(x.shape[0], -1).cpu().numpy().view(np.uint8)
            else:
                return x.reshape(x.shape[0], -1).view(np.uint8)
        def deserialize_transition(x):
            # x: [num_envs, flattened_data_length]
            nonlocal shape, dtype
            if shape is None or dtype is None:
                raise ValueError(f'Must call serialization function at least once with a sample {name} to infer the required shape and dtype for the {name} deserialization. Alternatively, you can provide the shape and dtype as arguments to `make_default_serde`.')
            if isinstance(dtype, torch.dtype):
                # Note: the `copy() is necessary because the data isn't contiguous in memory.
                return torch.from_numpy(x).view(dtype).view(*shape[1:])
            else:
                return x.view(dtype).reshape(shape[1:])
        def deserialize_trajectory(x):
            nonlocal shape, dtype
            if shape is None or dtype is None:
                raise ValueError(f'Must call serialization function at least once with a sample {name} to infer the required shape and dtype for the {name} deserialization. Alternatively, you can provide the shape and dtype as arguments to `make_default_serde`.')
            # Two cases:
            # x: [batch, trajectory length, flattened_data_length]
            # x: [trajectory length, flattened_data_length]
            if len(x.shape) == 2:
                if isinstance(dtype, torch.dtype):
                    return torch.from_numpy(x).view(dtype).view(-1, *shape[1:])
                else:
                    return x.view(dtype).reshape(-1, *shape[1:])
            if len(x.shape) == 3:
                if isinstance(dtype, torch.dtype):
                    return torch.from_numpy(x).view(dtype).view(x.shape[0], x.shape[1], *shape[1:])
                else:
                    return x.view(dtype).reshape(x.shape[0], x.shape[1], *shape[1:])
        return serialize, deserialize_transition, deserialize_trajectory
    obs = make('obs')
    reward = make('reward')
    action = make('action')
    serialize_fn = SerializeFn(
        obs=obs[0],
        reward=reward[0],
        misc=lambda _: np.empty([1,0], dtype=np.uint8),
        action=action[0],
    )
    deserialize_transition_fn = SerializeFn(
        obs=obs[1],
        reward=reward[1],
        misc=lambda x: None,
        action=action[1],
    )
    deserialize_trajectory_fn = SerializeFn(
        obs=obs[2],
        reward=reward[2],
        misc=lambda x: None,
        action=action[2],
    )
    return serialize_fn, deserialize_transition_fn, deserialize_trajectory_fn

class DataSizes(NamedTuple):
    obs: int
    reward: int
    misc: int
    action: int

class DataSlices(NamedTuple):
    obs: slice
    reward: slice
    terminated: slice
    truncated: slice
    misc: slice
    action: slice


class NumpyBackedVecHistoryBuffer():
    __slots__ = ( 'max_len', 'num_envs', 'data_size', 'serialize_fn', 'deserialize_transition_fn', 'deserialize_trajectory_fn', 'trajectory_length', 'device', '_insert_index', '_incomplete_row', '_num_rows', '_transition_view', '_trajectory_view', 'obs_history', 'action_history', 'reward_history', 'terminated_history', 'truncated_history', 'misc_history' )
    def __init__(self,
                 max_len: int,
                 num_envs: int,
                 data_size: DataSizes | None = None,
                 serialize_fn: SerializeFn | None = None,
                 deserialize_transition_fn: SerializeFn | None = None,
                 deserialize_trajectory_fn: SerializeFn | None = None,
                 trajectory_length: int | None = None,
                 device: torch.device = torch.device('cpu')
                 ) -> None:
        # Validate input
        if max_len < 1:
            raise ValueError('`max_len` must be at least 1')
        if num_envs < 1:
            raise ValueError('`num_envs` must be at least 1')

        self.max_len = max_len
        self.num_envs = num_envs
        self.data_size = data_size
        self.serialize_fn = serialize_fn
        self.deserialize_trajectory_fn = deserialize_trajectory_fn
        self.deserialize_transition_fn = deserialize_transition_fn
        self.trajectory_length = trajectory_length
        self.device = device

        if self.serialize_fn is None and self.deserialize_transition_fn is None and self.deserialize_trajectory_fn is None:
            default_serialize_fn, default_deserialize_transition_fn, default_deserialize_trajectory_fn = make_default_serde()
            self.serialize_fn = default_serialize_fn
            self.deserialize_transition_fn = default_deserialize_transition_fn
            self.deserialize_trajectory_fn = default_deserialize_trajectory_fn
        elif self.serialize_fn is not None and self.deserialize_transition_fn is not None and self.deserialize_trajectory_fn is not None:
            pass # All good
        elif self.serialize_fn is not None or self.deserialize_transition_fn is not None or self.deserialize_trajectory_fn is not None:
            raise ValueError(f'Must provide all or none of `serialize_fn`, `deserialize_transition_fn`, and `deserialize_trajectory_fn`. `serialize_fn`: {self.serialize_fn is not None}, `deserialize_transition_fn`: {self.deserialize_transition_fn is not None}, `deserialize_trajectory_fn`: {self.deserialize_trajectory_fn is not None}')

        self._insert_index = 0
        self._incomplete_row = False
        self._num_rows = 0
        self._transition_view = TransitionView(self)
        self._trajectory_view = TrajectoryView(self)

        if self.data_size is None:
            self.obs_history = None
            self.action_history = None
            self.reward_history = None
            self.misc_history = None
        else:
            self.obs_history = np.zeros((max_len, num_envs, self.data_size.obs), dtype=np.uint8)
            self.action_history = np.zeros((max_len, num_envs, self.data_size.action), dtype=np.uint8)
            self.reward_history = np.zeros((max_len, num_envs, self.data_size.reward), dtype=np.uint8)
            self.misc_history = np.zeros((max_len, num_envs, self.data_size.misc), dtype=np.uint8)
        self.terminated_history = np.zeros((max_len, num_envs), dtype=bool)
        self.truncated_history = np.zeros((max_len, num_envs), dtype=bool)

        assert self.deserialize_trajectory_fn is not None, 'Deserialization function for trajectories must be provided'
        self._transition_view = TransitionView(self)
        self._trajectory_view = NumpyBackedTrajectoryView(self, self.deserialize_trajectory_fn)

    def _append_obs_history(self, index, value):
        assert self.serialize_fn is not None, 'Serialization function must be provided to append data'
        serialized = self.serialize_fn.obs(value)
        if self.obs_history is None:
            self.obs_history = np.zeros((self.max_len, self.num_envs, serialized.shape[1]), dtype=np.uint8)
        self.obs_history[index,:] = serialized

    def _append_action_history(self, index, value):
        assert self.serialize_fn is not None, 'Serialization function must be provided to append data'
        serialized = self.serialize_fn.action(value)
        if self.action_history is None:
            self.action_history = np.zeros((self.max_len, self.num_envs, serialized.shape[1]), dtype=np.uint8)
        self.action_history[index,:] = serialized

    def _append_reward_history(self, index, value):
        assert self.serialize_fn is not None, 'Serialization function must be provided to append data'
        serialized = self.serialize_fn.reward(value)
        if self.reward_history is None:
            self.reward_history = np.zeros((self.max_len, self.num_envs, serialized.shape[1]), dtype=np.uint8)
        self.reward_history[index,:] = serialized

    def _append_misc_history(self, index, value):
        assert self.serialize_fn is not None, 'Serialization function must be provided to append data'
        serialized = self.serialize_fn.misc(value)
        if self.misc_history is None:
            self.misc_history = np.zeros((self.max_len, self.num_envs, serialized.shape[1]), dtype=np.uint8)
        self.misc_history[index,:] = serialized

    def append_obs(self,
                   obs,
                   reward = None,
                   terminated = None,
                   truncated = None,
                   misc = None,
                   ) -> None:
        assert not self._incomplete_row, 'Cannot append observation before appending action'
        assert self.serialize_fn is not None, 'Serialization function must be provided to append data'

        #self.obs_history[self._insert_index,:] = self.serialize_fn.obs(obs)
        self._append_obs_history(self._insert_index, obs)
        if reward is not None: # If reward is not None, then terminated and truncated must also not be None
            #self.reward_history[self._insert_index,:] = self.serialize_fn.reward(reward)
            self._append_reward_history(self._insert_index, reward)
            self.terminated_history[self._insert_index,:] = terminated
            self.truncated_history[self._insert_index,:] = truncated
        #self.misc_history[self._insert_index,:] = self.serialize_fn.misc(misc)
        self._append_misc_history(self._insert_index, misc)

        self._incomplete_row = True

    def append_action(self, action):
        assert self._incomplete_row, 'Cannot append action before appending observation'
        assert self.serialize_fn is not None, 'Serialization function must be provided to append data'

        #self.action_history[self._insert_index,:] = self.serialize_fn.action(action)
        self._append_action_history(self._insert_index, action)
        self._incomplete_row = False

        self._insert_index = (self._insert_index + 1) % self.max_len
        self._num_rows = min(self._num_rows + 1, self.max_len)

    def clear(self, fullclear: bool = False) -> None:
        if fullclear:
            self._insert_index = 0
            self._incomplete_row = False
            self._num_rows = 0
        else:
            last_row_index = (self._insert_index - 1) % self.max_len

            self.obs_history[0,:] = self.obs_history[last_row_index,:]
            self.action_history[0,:] = self.action_history[last_row_index,:]
            self.reward_history[0,:] = self.reward_history[last_row_index,:]
            self.terminated_history[0,:] = self.terminated_history[last_row_index,:]
            self.truncated_history[0,:] = self.truncated_history[last_row_index,:]
            self.misc_history[0,:] = self.misc_history[last_row_index,:]

            self._insert_index = 1
            self._incomplete_row = False
            self._num_rows = 1

    def _get_dones(self):
        """ Get a boolean array of shape (num_rows, num_envs) indicating which transitions are done (terminated or truncated). This array is shifted such that the value at index i corresponds to the i-th oldest transition. """
        i = (self._insert_index - 1) % self.max_len
        done = self.terminated_history[:i,:] | self.truncated_history[:i,:]
        if self._num_rows == self.max_len:
            done2 = self.terminated_history[i+1:,:] | self.truncated_history[i+1:,:]
            done = np.concatenate([done2, done], axis=0)
        return done

    @property
    def num_transitions(self):
        if self._num_rows == 0:
            return 0
        done = self._get_dones()
        return self.num_envs * (self._num_rows - 1) - done.sum()

    def get_transition(self, index) -> Transition:
        if index >= self.num_transitions:
            raise IndexError('Index out of range')
        assert self.deserialize_transition_fn is not None
        assert self.obs_history is not None
        assert self.action_history is not None
        assert self.reward_history is not None
        assert self.misc_history is not None

        # Determine which envionment we're selecting from, and the index of the transition within that environment
        done = self._get_dones()
        num_transitions = np.full(self.num_envs, fill_value=(self._num_rows - 1), dtype=int) - np.sum(done, axis=0).flatten()
        env_index = 0
        idx_in_env = index
        for n in num_transitions:
            if idx_in_env >= n:
                idx_in_env -= n
                env_index += 1

        # Adjust the transition index for terminations
        i = 0
        for i,d in enumerate(done[:,env_index]):
            if d:
                idx_in_env += 1
            if i == idx_in_env:
                break
        if self._num_rows == self.max_len:
            i = (i + self._insert_index) % self.max_len
        i0 = i
        i1 = (i + 1) % self.max_len
        e = slice(env_index, env_index+1)

        return Transition(
            self.deserialize_transition_fn.obs(self.obs_history[i0,e]),
            self.deserialize_transition_fn.action(self.action_history[i0,e]),
            self.deserialize_transition_fn.obs(self.obs_history[i1,e]),
            self.deserialize_transition_fn.reward(self.reward_history[i1,e]),
            self.terminated_history[i1,e].item(),
            self.truncated_history[i1,e].item(),
            self.deserialize_transition_fn.misc(self.misc_history[i0,e]),
            self.deserialize_transition_fn.misc(self.misc_history[i1,e]),
        )

    @property
    def num_trajectories(self):
        if self.trajectory_length is None:
            raise ValueError('Trajectory sampling is disabled. To enable, set `trajectory_length` to a positive integer.')
        num_obs_per_env = self._num_rows
        if num_obs_per_env < self.trajectory_length:
            return 0
        return (num_obs_per_env - self.trajectory_length) * self.num_envs

    def get_trajectory(self, index) -> Trajectory:
        if index >= self.num_trajectories:
            raise IndexError('Index out of range')
        assert self.trajectory_length is not None
        assert self.deserialize_trajectory_fn is not None

        num_obs_per_env = self._num_rows
        trajectories_per_env = num_obs_per_env - self.trajectory_length
        env_index = index // trajectories_per_env
        step_index = index % trajectories_per_env
        if self._num_rows == self.max_len:
            step_index = (step_index + self._insert_index) % self.max_len

        e = env_index
        s0 = slice(step_index, step_index+self.trajectory_length)
        s1 = slice(step_index+1, step_index+self.trajectory_length+1)

        def take_slice_wrap(arr, s):
            if s.stop > self.max_len:
                return np.concatenate([arr[s, e], arr[:s.stop % self.max_len, e]], axis=0)
            else:
                return arr[s, e]

        return Trajectory(
            obs = self.deserialize_trajectory_fn.obs(take_slice_wrap(self.obs_history, s0)),
            action = self.deserialize_trajectory_fn.action(take_slice_wrap(self.action_history, s0)),
            next_obs = self.deserialize_trajectory_fn.obs(take_slice_wrap(self.obs_history, s1)),
            reward = self.deserialize_trajectory_fn.reward(take_slice_wrap(self.reward_history, s1)),
            terminated = torch.from_numpy(take_slice_wrap(self.terminated_history, s1)).flatten(),
            truncated = torch.from_numpy(take_slice_wrap(self.truncated_history, s1)).flatten(),
            misc = self.deserialize_trajectory_fn.misc(take_slice_wrap(self.misc_history, s0)),
            next_misc = self.deserialize_trajectory_fn.misc(take_slice_wrap(self.misc_history, s1)),
        )

    def get_random_trajectory(self):
        index = np.random.randint(0, self.num_trajectories)
        return self.get_trajectory(index)

    def get_random_batch_serialized_trajectory(self, batch_size: int, replacement=False):
        assert self.trajectory_length is not None
        assert self.deserialize_trajectory_fn is not None

        indices = np.random.choice(self.num_trajectories, size=batch_size, replace=replacement)
        slices = []
        for index in indices:
            num_obs_per_env = self._num_rows
            trajectories_per_env = num_obs_per_env - self.trajectory_length
            env_index = index // trajectories_per_env
            step_index = index % trajectories_per_env
            if self._num_rows == self.max_len:
                step_index = (step_index + self._insert_index) % self.max_len

            e = env_index
            s0 = slice(step_index, step_index+self.trajectory_length)
            s1 = slice(step_index+1, step_index+self.trajectory_length+1)

            slices.append((e, s0, s1))

        def take_slice_wrap(arr, e, s):
            if s.stop > self.max_len:
                return np.concatenate([arr[s, e], arr[:s.stop % self.max_len, e]], axis=0)
            else:
                return arr[s, e]

        obs = self.deserialize_trajectory_fn.obs(
            np.stack([
                take_slice_wrap(self.obs_history, e, s0)
                for e, s0, s1 in slices
            ])
        )
        action = self.deserialize_trajectory_fn.action(
            np.stack([
                take_slice_wrap(self.action_history, e, s0)
                for e, s0, s1 in slices
            ])
        )
        next_obs = self.deserialize_trajectory_fn.obs(
            np.stack([
                take_slice_wrap(self.obs_history, e, s1)
                for e, s0, s1 in slices
            ])
        )
        reward = self.deserialize_trajectory_fn.reward(
            np.stack([
                take_slice_wrap(self.reward_history, e, s1)
                for e, s0, s1 in slices
            ])
        )
        terminated = torch.from_numpy(np.stack([
            take_slice_wrap(self.terminated_history, e, s1)
            for e, s0, s1 in slices
        ]))
        truncated = torch.from_numpy(np.stack([
            take_slice_wrap(self.truncated_history, e, s1)
            for e, s0, s1 in slices
        ]))
        misc = self.deserialize_trajectory_fn.misc(
            np.stack([
                take_slice_wrap(self.misc_history, e, s0)
                for e, s0, s1 in slices
            ])
        )
        next_misc = self.deserialize_trajectory_fn.misc(
            np.stack([
                take_slice_wrap(self.misc_history, e, s1)
                for e, s0, s1 in slices
            ])
        )

        return Trajectory(
            obs = torch.from_numpy(obs).to(self.device),
            action = torch.from_numpy(action).to(self.device),
            next_obs = torch.from_numpy(next_obs).to(self.device),
            reward = torch.from_numpy(reward).to(self.device),
            terminated = terminated.to(self.device),
            truncated = truncated.to(self.device),
            misc = misc,
            next_misc = next_misc,
        )

    @property
    def transitions(self):
        return self._transition_view

    @property
    def trajectories(self):
        return self._trajectory_view

    def state_dict(self):
        return {
            '_insert_index': self._insert_index,
            '_incomplete_row': self._incomplete_row,
            '_num_rows': self._num_rows,
            'obs_history': self.obs_history,
            'action_history': self.action_history,
            'reward_history': self.reward_history,
            'terminated_history': self.terminated_history,
            'truncated_history': self.truncated_history,
            'misc_history': self.misc_history,
        }

    def load_state_dict(self, state_dict):
        self._insert_index = state_dict['_insert_index']
        self._incomplete_row = state_dict['_incomplete_row']
        self._num_rows = state_dict['_num_rows']
        self.obs_history = state_dict['obs_history']
        self.action_history = state_dict['action_history']
        self.reward_history = state_dict['reward_history']
        self.terminated_history = state_dict['terminated_history']
        self.truncated_history = state_dict['truncated_history']
        self.misc_history = state_dict['misc_history']


VecHistoryBuffer = ListBackedVecHistoryBuffer
