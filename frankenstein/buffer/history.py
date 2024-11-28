import torch
from torch.utils.data.dataloader import default_collate
from torchtyping import TensorType
from tensordict import TensorDict

from .views import TrajectoryView, TransitionView, Trajectory, Transition


class HistoryBuffer():
    def __init__(self,
                 max_len: int,
                 trajectory_length: int | None = None,
                 batch_first: bool = False,
                 device: torch.device = torch.device('cpu')
                 ) -> None:
        """
        Args:
            max_len (int): The total number of observations to store. Note that this is different from the number of transitions stored, which can vary depending on the number of terminal states encountered.
            trajectory_length (int): Length of trajectories (number of transitions) to be sampled from this buffer. If `None`, then trajectory sampling is disabled. Note that one episode terminating into a new episode is considered a transition for the purposes of a trajectory.
            default_action: Action to use as padding at the end of an episode.
            batch_first (bool): If set to `True`, then all returned tensors will have the shape `(batch_size, seq_len, ...)`. Otherwise, they will take the shape `(seq_len, batch_size, ...)`. Default: `False`.
        """
        self.device = device
        self.max_len = max_len
        self.trajectory_length = trajectory_length
        self.batch_first = batch_first

        # Validate input
        if max_len < 1:
            # We enforce this because it makes the append_action method easier to implement. I don't think there's a good use case for allowing it.
            raise ValueError('`max_len` must be at least 1')

        # History is saved as a list instead of tensors to save on memory. This allows for data to be stored in formats like LazyFrames.
        self.obs_history: list[TensorType | TensorDict] = []
        self.action_history: list[TensorType | TensorDict] = []
        self.reward_history: list[float] = []
        self.terminated_history: list[bool] = []
        self.truncated_history: list[bool] = []
        self.misc_history: list = []

        self.default_reward = 0

        self._terminated_eps = False
        self._num_transitions = 0
        self._transition_view = TransitionView(self)
        self._trajectory_view = TrajectoryView(self)

    def append_obs(self,
                   obs: TensorType | TensorDict,
                   reward: float | None = None,
                   terminated: bool = False,
                   truncated: bool = False,
                   misc = None,
                   ) -> None:
        # Make sure the observations and actions are in sync
        assert len(self.obs_history) == len(self.action_history)

        # Handle boundary between episodes
        if reward is None:
            reward = self.default_reward
        else:
            self._num_transitions += 1

        # Save data
        self.obs_history.append(obs)
        self.reward_history.append(reward)
        self.terminated_history.append(terminated)
        self.truncated_history.append(truncated)
        self.misc_history.append(misc)

        if terminated or truncated:
            # The last episode just finished, so we receive a new observation to start the episode without an action in between.
            # Add an arbitrary action to pad the actions list.
            self.append_action(self.action_history[0])

        # Enforce max length
        if self.max_len is not None and len(self.obs_history) > self.max_len:
            if not self.terminated_history[0] and not self.truncated_history[0]:
                self._num_transitions -= 1
            self.obs_history = self.obs_history[1:]
            self.reward_history = self.reward_history[1:]
            self.terminated_history = self.terminated_history[1:]
            self.truncated_history = self.truncated_history[1:]
            self.action_history = self.action_history[1:]
            self.misc_history = self.misc_history[1:]

    def append_action(self, action):
        # Append action
        obs_history = self.obs_history
        action_history = self.action_history
        assert len(obs_history) == len(action_history)+1
        action_history.append(action)

    def clear(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_observations

    @property
    def num_observations(self):
        return len(self.obs_history)

    @property
    def num_transitions(self):
        return self._num_transitions

    def get_transition(self, index) -> Transition:
        if index >= self._num_transitions:
            raise IndexError('Index out of range')
        i = 0
        for i,(term,trunc) in enumerate(zip(self.terminated_history, self.truncated_history)):
            done = term or trunc
            if done:
                index += 1
            if i == index:
                break
        return Transition(
            self.obs_history[i],
            self.action_history[i],
            self.obs_history[i+1],
            self.reward_history[i+1],
            self.terminated_history[i+1],
            self.truncated_history[i+1],
            self.misc_history[i],
            self.misc_history[i+1]
        )

    @property
    def num_trajectories(self):
        if self.trajectory_length is None:
            raise ValueError('Trajectory sampling is disabled. To enable, set `trajectory_length` to a positive integer.')
        if self.num_observations < self.trajectory_length:
            return 0
        return self.num_observations - self.trajectory_length

    def get_trajectory(self, index) -> Trajectory:
        if index >= self.num_trajectories:
            raise IndexError('Index out of range')
        return Trajectory(
            obs = default_collate(self.obs_history[index:index+self.trajectory_length]),
            action = default_collate(self.action_history[index:index+self.trajectory_length]),
            next_obs = default_collate(self.obs_history[index+1:index+self.trajectory_length+1]),
            reward = torch.tensor(self.reward_history[index+1:index+self.trajectory_length+1]),
            terminated = torch.tensor(self.terminated_history[index+1:index+self.trajectory_length+1]),
            truncated = torch.tensor(self.truncated_history[index+1:index+self.trajectory_length+1]),
            misc = self.misc_history[index:index+self.trajectory_length],
            next_misc = self.misc_history[index+1:index+self.trajectory_length+1]
        )

    @property
    def transitions(self):
        return self._transition_view

    @property
    def trajectories(self):
        return self._trajectory_view


