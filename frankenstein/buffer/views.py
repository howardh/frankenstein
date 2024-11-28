from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, Mapping, List, NamedTuple, Any

import torch
from torchtyping import TensorType
from torch.utils.data.dataloader import default_collate
from tensordict import TensorDict


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
    reward: TensorType[float]
    """ Reward received after taking the action (`action`) at the start state (`obs`) """
    terminated: TensorType[bool]
    """ Whether the next state (`next_obs`) is terminal"""
    truncated: TensorType[bool]
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
    reward: TensorType[float]
    terminated: TensorType[bool]
    truncated: TensorType[bool]
    misc: list[Any]
    next_misc: list[Any]


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
