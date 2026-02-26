from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from jaxtyping import Float, Bool
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tensordict import TensorDict

if TYPE_CHECKING: # Avoid circular import problems
    from frankenstein.buffer.vec_history import NumpyBackedVecHistoryBuffer, SerializeFn


@dataclass
class Transition():
    obs: torch.Tensor | TensorDict
    """ Start state """
    action: torch.Tensor | TensorDict
    """ Action taken at the start state (`obs`) """
    next_obs: torch.Tensor | TensorDict
    """ State observed after taking the action (`action`) at the start state (`obs`) """
    reward: float
    """ Reward received after taking the action (`action`) at the start state (`obs`) """
    terminated: bool
    """ Whether the next state (`next_obs`) is terminal"""
    truncated: bool
    """ Whether the next state (`next_obs`) is terminal"""
    misc: Any
    """ Miscellaneous data associated with `obs` """
    next_misc: Any
    """ Miscellaneous data associated with `next_obs` """


@dataclass
class TransitionBatch():
    obs: torch.Tensor | TensorDict
    """ Start state """
    action: torch.Tensor | TensorDict
    """ Action taken at the start state (`obs`) """
    next_obs: torch.Tensor | TensorDict
    """ State observed after taking the action (`action`) at the start state (`obs`) """
    reward: Float[torch.Tensor, '...']
    """ Reward received after taking the action (`action`) at the start state (`obs`) """
    terminated: Bool[torch.Tensor, '...']
    """ Whether the next state (`next_obs`) is terminal"""
    truncated: Bool[torch.Tensor, '...']
    """ Whether the next state (`next_obs`) is terminal"""
    misc: Sequence[Any]
    """ Miscellaneous data associated with `obs` """
    next_misc: Sequence[Any]
    """ Miscellaneous data associated with `next_obs` """


@dataclass
class Trajectory():
    obs: torch.Tensor | TensorDict
    action: torch.Tensor | TensorDict
    next_obs: torch.Tensor | TensorDict
    reward: Float[torch.Tensor, '...']
    terminated: Bool[torch.Tensor, '...']
    truncated: Bool[torch.Tensor, '...']
    misc: list[Any]
    next_misc: list[Any]


@dataclass
class TrajectoryBatch(TransitionBatch):
    ... # Same as TransitionBatch


class TransitionView():
    def __init__(self, buffer):
        self._buffer = buffer

    def __len__(self):
        return self._buffer.num_transitions

    def __getitem__(self, index):
        return self._buffer.get_transition(index)

    def sample_batch(self, batch_size: int, replacement=True):
        device = self._buffer.device
        if replacement:
            transitions = [self._buffer.get_random_transition() for _ in range(batch_size)]
        else:
            # FIXME: This is 100x slower than sampling with replacement with VecHistoryBuffer.
            indices = torch.randint(0, len(self), (batch_size,))
            transitions = [self[i] for i in indices]
        return TransitionBatch(
            obs = default_collate([t.obs for t in transitions]).to(device),
            action = default_collate([t.action for t in transitions]).to(device),
            next_obs = default_collate([t.next_obs for t in transitions]).to(device),
            reward = torch.tensor([[t.reward] for t in transitions], dtype=torch.float32).to(device),
            terminated = torch.tensor([[t.terminated] for t in transitions]).to(device),
            truncated = torch.tensor([[t.truncated] for t in transitions]).to(device),
            misc = [t.misc for t in transitions],
            next_misc = [t.next_misc for t in transitions],
        )

    def sample_batch2(self, batch_size: int):
        device = self._buffer.device
        indices = torch.randint(0, len(self), (batch_size,))
        transitions = [self[i] for i in indices]
        return TransitionBatch(
            obs = default_collate([t.obs for t in transitions]).to(device),
            action = default_collate([t.action for t in transitions]).to(device),
            next_obs = default_collate([t.next_obs for t in transitions]).to(device),
            reward = torch.tensor([[t.reward] for t in transitions], dtype=torch.float32).to(device),
            terminated = torch.tensor([[t.terminated] for t in transitions]).to(device),
            truncated = torch.tensor([[t.truncated] for t in transitions]).to(device),
            misc = [t.misc for t in transitions],
            next_misc = [t.next_misc for t in transitions],
        )


class TrajectoryView():
    def __init__(self, buffer):
        self._buffer = buffer

    def __len__(self):
        return self._buffer.num_trajectories

    def __getitem__(self, index):
        return self._buffer.get_trajectory(index)

    def sample_batch(self, batch_size: int, replacement=True):
        device = self._buffer.device
        if replacement:
            trajectories = [self._buffer.get_random_trajectory() for _ in range(batch_size)]
        else:
            # XXX: Is this also as slow as the TransitionView version?
            indices = torch.randint(0, len(self), (batch_size,))
            trajectories = [self[i] for i in indices]
        return TrajectoryBatch(
            obs = default_collate([t.obs for t in trajectories]).to(device),
            action = default_collate([t.action for t in trajectories]).to(device),
            next_obs = default_collate([t.next_obs for t in trajectories]).to(device),
            reward = torch.stack([t.reward for t in trajectories]).to(device),
            terminated = torch.stack([t.terminated for t in trajectories]).to(device),
            truncated = torch.stack([t.truncated for t in trajectories]).to(device),
            misc = [t.misc for t in trajectories],
            next_misc = [t.next_misc for t in trajectories],
        )


class NumpyBackedTrajectoryView():
    def __init__(self, buffer: 'NumpyBackedVecHistoryBuffer', deserialize_trajectory_fn: 'SerializeFn'):
        self._buffer = buffer
        self._deserialize_trajectory_fn = deserialize_trajectory_fn

    def __len__(self):
        return self._buffer.num_trajectories

    def __getitem__(self, index):
        return self._buffer.get_trajectory(index)

    def sample_batch(self, batch_size: int, replacement=True):
        return self._buffer.get_random_batch_serialized_trajectory(batch_size, replacement)
