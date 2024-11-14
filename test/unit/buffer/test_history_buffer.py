import pytest
import torch
import numpy as np

from frankenstein.buffer.history import HistoryBuffer as Buffer


def test_max_len_0():
    """ An empty buffer should raise an error """
    with pytest.raises(ValueError):
        Buffer(max_len=0)


##################################################
# Transition sampling


def test_0_transition():
    buffer = Buffer(
        max_len=2,
    )

    assert len(buffer.transitions) == 0


def test_1_transition():
    buffer = Buffer(
        max_len=2,
    )

    buffer.append_obs(obs=1)
    buffer.append_action(1)
    buffer.append_obs(obs=2, reward=1, terminal=False)

    assert len(buffer.transitions) == 1

    transition = buffer.transitions[0]

    assert transition.obs == 1
    assert transition.action == 1
    assert transition.reward == 1
    assert transition.terminal is False
    assert transition.next_obs == 2


def test_1_transition_out_of_bounds():
    buffer = Buffer(
        max_len=2,
    )

    buffer.append_obs(obs=1)
    buffer.append_action(1)
    buffer.append_obs(obs=2, reward=1, terminal=False)

    assert len(buffer.transitions) == 1

    with pytest.raises(IndexError):
        buffer.transitions[1]


def test_2_transition():
    buffer = Buffer(
        max_len=10,
    )

    buffer.append_obs(obs=1)
    buffer.append_action(1)
    buffer.append_obs(obs=2, reward=1, terminal=False)
    buffer.append_action(2)
    buffer.append_obs(obs=3, reward=2, terminal=False)

    assert len(buffer.transitions) == 2

    transition = buffer.transitions[0]
    assert transition.obs == 1
    assert transition.action == 1
    assert transition.reward == 1
    assert transition.terminal is False
    assert transition.next_obs == 2

    transition = buffer.transitions[1]
    assert transition.obs == 2
    assert transition.action == 2
    assert transition.reward == 2
    assert transition.terminal is False
    assert transition.next_obs == 3


def test_2_transition_overflow():
    """ Test that when the buffer overflows and overwrites old data, it does so correctly. """
    buffer = Buffer(
        max_len=2,
    )

    buffer.append_obs(obs=1)
    buffer.append_action(1)
    buffer.append_obs(obs=2, reward=1, terminal=False)
    buffer.append_action(2)
    buffer.append_obs(obs=3, reward=2, terminal=False)

    assert len(buffer.transitions) == 1

    transition = buffer.transitions[0]
    assert transition.obs == 2
    assert transition.action == 2
    assert transition.reward == 2
    assert transition.terminal is False
    assert transition.next_obs == 3


def test_transition_with_termination():
    """ Test that when an episode terminates, we can correctly retrieve the terminal step. """
    buffer = Buffer(
        max_len=4,
    )

    buffer.append_obs(obs=1)
    buffer.append_action(1)
    buffer.append_obs(obs=2, reward=1, terminal=True)

    buffer.append_obs(obs=3)
    buffer.append_action(2)
    buffer.append_obs(obs=4, reward=2, terminal=False)

    assert len(buffer.transitions) == 2

    transition = buffer.transitions[0]
    assert transition.obs == 1
    assert transition.action == 1
    assert transition.reward == 1
    assert transition.terminal is True
    assert transition.next_obs == 2

    transition = buffer.transitions[1]
    assert transition.obs == 3
    assert transition.action == 2
    assert transition.reward == 2
    assert transition.terminal is False
    assert transition.next_obs == 4

##################################################
# Trajectory sampling

# TODO: Length 1 trajectories. Use the same tests as for transitions.

# TODO: Longer trajectories.


def test_trajectory_too_few_transitions():
    """ When there are too few transitions in the buffer, we should see no trajectories available. """
    buffer = Buffer(
        max_len=10,
        trajectory_length=3,
    )

    assert len(buffer.trajectories) == 0

    buffer.append_obs(1)
    buffer.append_action(1)

    assert len(buffer.trajectories) == 0

    buffer.append_obs(1,0,False)
    buffer.append_action(1)

    assert len(buffer.trajectories) == 0

    buffer.append_obs(1,0,False)
    buffer.append_action(1)

    assert len(buffer.trajectories) == 0

    buffer.append_obs(1,0,False)
    buffer.append_action(1)

    assert len(buffer.trajectories) == 1


def test_trajectory_indexing():
    buffer = Buffer(
        max_len=10,
        trajectory_length=3,
    )

    buffer.append_obs(0)
    for i in range(9):
        buffer.append_action(i)
        buffer.append_obs(obs=i+1, reward=i+0.1, terminal=False)

    traj = buffer.trajectories[0]
    assert traj.obs == [0,1,2]
    assert traj.next_obs == [1,2,3]
    assert traj.action == [0,1,2]
    assert traj.reward == [0.1,1.1,2.1]

    traj = buffer.trajectories[1]
    assert traj.obs == [1,2,3]
    assert traj.next_obs == [2,3,4]
    assert traj.action == [1,2,3]
    assert traj.reward == [1.1,2.1,3.1]


def test_trajectory_out_of_bounds():
    buffer = Buffer(
        max_len=10,
        trajectory_length=3,
    )

    buffer.append_obs(0)
    for i in range(3):
        buffer.append_action(i)
        buffer.append_obs(obs=i+1, reward=i+0.1, terminal=False)

    buffer.trajectories[0]
    with pytest.raises(IndexError):
        buffer.trajectories[1]
