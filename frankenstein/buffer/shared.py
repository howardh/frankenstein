from typing import Generic, TypeVar, Mapping, Sequence, Tuple, List
import threading

import torch
from torch import multiprocessing as mp
from torch.utils.data.dataloader import default_collate

ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')
MiscType = TypeVar('MiscType')


def make_buffer(space, rollout_length):
    if space is None:
        return None
    if space['type'] == 'box':
        return torch.zeros(
                size=(rollout_length + 1, *space['shape']),
                dtype=space.get('dtype', torch.float),
        ).share_memory_()
    elif space['type'] == 'discrete':
        return torch.zeros(
                size=(rollout_length + 1, 1),
                dtype=space.get('dtype', torch.uint8),
        ).share_memory_()
    elif space['type'] == 'dict':
        return {
            k: make_buffer(v, rollout_length) for k,v in space['data'].items()
        }
    elif space['type'] == 'tuple':
        return tuple(
            make_buffer(v, rollout_length) for v in space['data']
        )
    else: # pragma: no cover
        raise NotImplementedError(
                f'Unsupported observation space of type {space["type"]}')


def copy_tensor(src, dest, dest_indices=...):
    """ Copy the observation from src to dest. """
    if isinstance(dest, torch.Tensor):
        if isinstance(src, torch.Tensor):
            dest.__setitem__(dest_indices, src.to(dtype=dest.dtype))
        else:
            try:
                dest.__setitem__(dest_indices, torch.tensor(src, dtype=dest.dtype))
            except:
                raise TypeError(f"Unable to convert object of type {type(src)} to tensor")
    elif isinstance(dest, Mapping):
        for k in src.keys():
            copy_tensor(src[k], dest[k], dest_indices=dest_indices)
    elif isinstance(dest, Sequence):
        for s,d in zip(src, dest):
            copy_tensor(s, d, dest_indices=dest_indices)
    elif dest is None:
        pass
    else: # pragma: no cover
        raise NotImplementedError(
                f'Unsupported destination type {type(dest)}')


def transpose(x, dim0, dim1):
    if isinstance(x, torch.Tensor):
        return x.transpose(dim0, dim1)
    elif isinstance(x, Mapping):
        return {k: transpose(v, dim0, dim1) for k,v in x.items()}
    elif isinstance(x, Tuple):
        return tuple(transpose(v, dim0, dim1) for v in x)
    elif isinstance(x, List):
        return list(transpose(v, dim0, dim1) for v in x)
    else: # pragma: no cover
        raise NotImplementedError(f"Unknown data type: {type(x)}")


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, Mapping):
        return {k: to_device(v, device) for k,v in x.items()}
    elif isinstance(x, Tuple):
        return tuple(to_device(v, device) for v in x)
    elif isinstance(x, List):
        return list(to_device(v, device) for v in x)
    else: # pragma: no cover
        raise NotImplementedError(f"Unknown data type: {type(x)}")


class SharedBuffer(Generic[ObsType, ActionType, MiscType]):
    """
    Buffer for data to be shared between processes.
    """
    def __init__(self,
            num_buffers,
            rollout_length,
            observation_space,
            action_space,
            misc_space,
            mp_context=mp):
        self.num_buffers = num_buffers
        self.observation_space = observation_space
        self.action_space = action_space
        self.misc_space = misc_space

        # Initialize shared memory
        self.buffers = [
                SingleSharedBuffer(
                    rollout_length=rollout_length,
                    observation_space=observation_space,
                    action_space=action_space,
                    misc_space=misc_space,
                ) for _ in range(num_buffers)
        ]
        self._full_queue = mp_context.SimpleQueue()
        self._empty_queue = mp_context.SimpleQueue()

        for i in range(num_buffers):
            self._empty_queue.put(i)

        # Everything below is not shared between processes
        self.current_buffer_index = None
        self.last_obs = None
        self.last_action = None

    def get_batch(self, batch_size, device=torch.device('cpu'), lock=threading.Lock()):
        with lock:
            # Get the buffer index for this environment/worker.
            indices = [self._full_queue.get() for _ in range(batch_size)]
            return SharedBufferBatch(
                    buffers=self,
                    indices=indices,
                    device=device,
            )

    def append_obs(self, obs, reward, terminal, misc):
        # Get the buffer index for this environment/worker.
        buffer_index = self.current_buffer_index
        if buffer_index is None:
            buffer_index = self._empty_queue.get()
            self.current_buffer_index = buffer_index
            self.buffers[buffer_index].reset()
            if self.last_obs is not None:
                self.buffers[buffer_index].append_obs(*self.last_obs)
                self.buffers[buffer_index].append_action(self.last_action)

        # Save a copy of this observation
        self.last_obs = (obs, reward, terminal, misc)

        # Append the data to the buffer.
        self.buffers[buffer_index].append_obs(*self.last_obs)

        # If we've reached a terminal state, then we don't want to copy this observation to the next buffer
        if terminal:
            self.last_obs = None
            # Check if the buffer is full.
            if self.buffers[buffer_index].is_full():
                # Put the buffer index back in the empty queue.
                self._full_queue.put(buffer_index)
                self.current_buffer_index = None

    def append_action(self, action):
        # Get the buffer index for this environment/worker.
        buffer_index = self.current_buffer_index
        assert buffer_index is not None

        # Save a copy of this action
        self.last_action = action

        # Append the data to the buffer.
        self.buffers[buffer_index].append_action(action)

        # Check if the buffer is full.
        if self.buffers[buffer_index].is_full():
            # Put the buffer index back in the empty queue.
            self._full_queue.put(buffer_index)
            self.current_buffer_index = None


class SingleSharedBuffer(Generic[ObsType, ActionType, MiscType]):
    def __init__(self,
            rollout_length,
            observation_space,
            action_space,
            misc_space):
        self._rollout_length = rollout_length
        self.observation_space = observation_space
        self.action_space = action_space
        self.misc_space = misc_space

        self.obs_history = make_buffer(observation_space, rollout_length)
        self.action_history = make_buffer(action_space, rollout_length)
        self.reward_history = torch.zeros(
                size=(rollout_length+1, 1), dtype=torch.float32).share_memory_()
        self.terminal_history = torch.zeros(
                size=(rollout_length+1, 1), dtype=torch.bool).share_memory_()
        self.misc_history = make_buffer(misc_space, rollout_length)

        self._obs_index = 0
        self._action_index = 0

    def is_full(self):
        return self._obs_index >= self._rollout_length+1

    def reset(self):
        self._obs_index = 0
        self._action_index = 0

    def append_obs(self, obs, reward, terminal, misc):
        copy_tensor(dest=self.obs_history, src=obs, dest_indices=[self._obs_index])
        self.reward_history[self._obs_index] = reward
        self.terminal_history[self._obs_index] = terminal
        copy_tensor(dest=self.misc_history, src=misc, dest_indices=[self._obs_index])
        self._obs_index += 1
        if terminal:
            self._action_index += 1

    def append_action(self, action):
        copy_tensor(
                dest=self.action_history,
                src=action,
                dest_indices=[self._action_index])
        self._action_index += 1


class SharedBufferBatch(Generic[ObsType, ActionType, MiscType]):
    """
    Buffer for data to be shared between processes.
    """
    def __init__(self, buffers: SharedBuffer, device: torch.device, indices):
        self.buffers = buffers
        self.indices = indices

        def postprocess(x):
            x = default_collate(x)
            x = transpose(x, 0, 1)
            x = to_device(x, device)
            return x

        self.obs = postprocess(
                [self.buffers.buffers[i].obs_history for i in indices])
        self.action = postprocess(
                [self.buffers.buffers[i].action_history for i in indices])
        self.reward = postprocess(
                [self.buffers.buffers[i].reward_history for i in indices])
        self.terminal = postprocess(
                [self.buffers.buffers[i].terminal_history for i in indices])
        if self.buffers.buffers[0].misc_history is not None:
            self.misc = postprocess(
                    [self.buffers.buffers[i].misc_history for i in indices])
        else:
            self.misc = None

    def release(self):
        for i in self.indices:
            self.buffers._empty_queue.put(i)


