import pytest
import torch
import numpy as np

from frankenstein.buffer.vec_history import VecHistoryBuffer as Buffer


def test_max_len_0():
    with pytest.raises(Exception):
        Buffer(
            num_envs=1,
            max_len=0,
        )


def test_num_envs_0():
    with pytest.raises(Exception):
        Buffer(
            num_envs=0,
            max_len=1,
        )


def test_1_env():
    buffer = Buffer(
        num_envs=1,
        max_len=3,
    )
    buffer.append_obs(obs=np.array([[1, 2, 3]]))
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([[1, 2, 3]])+1, reward=np.array([0]), terminal=np.array([False]))
    buffer.append_action(action=np.array([1]))
    buffer.append_obs(
        obs=np.array([[1, 2, 3]])+2, reward=np.array([0]), terminal=np.array([False]))

    seq_len, batch_size, obs_len = buffer.obs.shape
    assert seq_len == 3
    assert batch_size == 1
    assert obs_len == 3
    assert (buffer.obs == torch.tensor([[[1, 2, 3]], [[2, 3, 4]], [[3, 4, 5]]])).all()

    seq_len, batch_size = buffer.reward.shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.reward == torch.tensor([[0], [0], [0]])).all()

    seq_len, batch_size = buffer.terminal.shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.terminal == torch.tensor([[False], [False], [False]])).all()

    seq_len, batch_size = buffer.action.shape
    assert seq_len == 2
    assert batch_size == 1
    assert (buffer.action == torch.tensor([[0], [1]])).all()


def test_episode_termination():
    buffer = Buffer(
        num_envs=1,
        max_len=10,
    )
    buffer.append_obs(obs=np.array([0]))
    buffer.append_action(action=np.array([10]))
    buffer.append_obs(obs=np.array([1]), reward=np.array([0]), terminal=np.array([False]))
    buffer.append_action(action=np.array([11]))
    buffer.append_obs(obs=np.array([2]), reward=np.array([0]), terminal=np.array([True]))
    buffer.append_action(action=np.array([5]))

    buffer.append_obs(obs=np.array([0]), reward=np.array([0]), terminal=np.array([False]))
    buffer.append_action(action=np.array([10]))
    buffer.append_obs(obs=np.array([1]), reward=np.array([0]), terminal=np.array([False]))
    buffer.append_action(action=np.array([11]))
    buffer.append_obs(obs=np.array([2]), reward=np.array([0]), terminal=np.array([True]))

    buffer.action == torch.tensor([[10], [11], [5], [10], [11]])
    buffer.terminal == torch.tensor([[False], [False], [True], [False], [False], [True]])


def test_exceed_max_len():
    buffer = Buffer(
        num_envs=1,
        max_len=1,
    )

    buffer.append_obs(obs=np.array([[1, 2, 3]]))
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([[1, 2, 3]])+1, reward=np.array([0]), terminal=np.array([False]))
    buffer.append_action(action=np.array([1]))
    buffer.append_obs(
        obs=np.array([[1, 2, 3]])+2, reward=np.array([0]), terminal=np.array([False]))

    seq_len, batch_size, obs_len = buffer.obs.shape
    assert seq_len == 2
    assert batch_size == 1
    assert obs_len == 3
    assert (buffer.obs == torch.tensor([[[2, 3, 4]], [[3, 4, 5]]])).all()

    seq_len, batch_size = buffer.reward.shape
    assert seq_len == 2
    assert batch_size == 1
    assert (buffer.reward == torch.tensor([[0], [0]])).all()

    seq_len, batch_size = buffer.terminal.shape
    assert seq_len == 2
    assert batch_size == 1
    assert (buffer.terminal == torch.tensor([[False], [False]])).all()


def test_clear():
    buffer = Buffer(
        num_envs=1,
        max_len=10,
    )
    buffer.append_obs(obs=np.array([0]))
    buffer.append_action(action=np.array([10]))
    buffer.append_obs(obs=np.array([1]), reward=np.array([0]), terminal=np.array([False]))
    buffer.append_action(action=np.array([11]))
    buffer.append_obs(obs=np.array([2]), reward=np.array([0]), terminal=np.array([True]))
    buffer.append_action(action=np.array([5]))
    buffer.append_obs(obs=np.array([0]))
    buffer.append_action(action=np.array([10]))
    buffer.append_obs(obs=np.array([1]), reward=np.array([0]), terminal=np.array([False]))
    buffer.append_action(action=np.array([11]))
    buffer.append_obs(obs=np.array([2]), reward=np.array([0]), terminal=np.array([True]))

    buffer.obs == torch.tensor([[0], [1], [2], [0], [1], [2]])
    buffer.action == torch.tensor([[10], [11], [5], [10], [11]])
    buffer.terminal == torch.tensor([[False], [False], [True], [False], [False], [True]])
    buffer.reward == torch.tensor([[0], [0], [0], [0], [0], [0]])

    buffer.clear()

    # It should keep the last observation/reward/terminal because it's part of the next transition. The action history should be cleared.
    buffer.obs == torch.tensor([[2]])
    buffer.action == torch.tensor([])
    buffer.terminal == torch.tensor([[True]])
    buffer.reward == torch.tensor([[0]])


def test_full_clear():
    buffer = Buffer(
        num_envs=1,
        max_len=10,
    )
    buffer.append_obs(obs=np.array([0]))
    buffer.append_action(action=np.array([10]))
    buffer.append_obs(obs=np.array([1]), reward=np.array([0]), terminal=np.array([False]))
    buffer.append_action(action=np.array([11]))
    buffer.append_obs(obs=np.array([2]), reward=np.array([0]), terminal=np.array([True]))
    buffer.append_action(action=np.array([5]))
    buffer.append_obs(obs=np.array([0]))
    buffer.append_action(action=np.array([10]))
    buffer.append_obs(obs=np.array([1]), reward=np.array([0]), terminal=np.array([False]))
    buffer.append_action(action=np.array([11]))
    buffer.append_obs(obs=np.array([2]), reward=np.array([0]), terminal=np.array([True]))

    buffer.obs == torch.tensor([[0], [1], [2], [0], [1], [2]])
    buffer.action == torch.tensor([[10], [11], [5], [10], [11]])
    buffer.terminal == torch.tensor([[False], [False], [True], [False], [False], [True]])
    buffer.reward == torch.tensor([[0], [0], [0], [0], [0], [0]])

    buffer.clear(fullclear=True)

    assert buffer.obs_history == []
    assert buffer.action_history == []
    assert buffer.reward_history == []
    assert buffer.terminal_history == []
    assert buffer.misc_history == []


# Testing various data types

def test_numpy_obs_int_action():
    buffer = Buffer(
        num_envs=2,
        max_len=3,
    )

    buffer.append_obs(obs=np.array([[1, 2, 3], [1, 2, 3]]))
    buffer.append_action(action=np.array([0, 0]))
    buffer.append_obs(
        obs=np.array([[1, 2, 3], [1, 2, 3]])+1, reward=np.array([0, 0]), terminal=np.array([False, False]))
    buffer.append_action(action=np.array([1, 1]))
    buffer.append_obs(
        obs=np.array([[1, 2, 3], [1, 2, 3]])+2, reward=np.array([0, 0]), terminal=np.array([False, False]))

    assert (buffer.obs[:, 0, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert (buffer.action[:, 0] == torch.tensor([0, 1])).all()
    assert (buffer.obs[:, 1, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert (buffer.action[:, 1] == torch.tensor([0, 1])).all()


def test_numpy_obs_numpy_action():
    buffer = Buffer(
        num_envs=2,
        max_len=3,
    )

    buffer.append_obs(obs=np.array([[1, 2, 3], [1, 2, 3]]))
    buffer.append_action(action=np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32))
    buffer.append_obs(
        obs=np.array([[1, 2, 3], [1, 2, 3]])+1, reward=np.array([0, 0]), terminal=np.array([False, False]))
    buffer.append_action(action=np.array([[0.2, 0.3], [0.2, 0.3]], dtype=np.float32))
    buffer.append_obs(
        obs=np.array([[1, 2, 3], [1, 2, 3]])+2, reward=np.array([0, 0]), terminal=np.array([False, False]))
    buffer.append_action(action=np.array([[0.3, 0.4], [0.3, 0.4]], dtype=np.float32))

    assert (buffer.obs[:, 0, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert torch.isclose(
        buffer.action[:, 0],
        torch.tensor([[.1, .2], [.2, .3], [.3, .4]])
    ).all()
    assert (buffer.obs[:, 1, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert torch.isclose(
        buffer.action[:, 1],
        torch.tensor([[.1, .2], [.2, .3], [.3, .4]])
    ).all()


def test_torch_obs_torch_action():
    buffer = Buffer(
        num_envs=2,
        max_len=3,
    )

    buffer.append_obs(obs=torch.tensor([[1, 2, 3], [1, 2, 3]]))
    buffer.append_action(action=torch.tensor([[0.1, 0.2], [0.1, 0.2]]))
    buffer.append_obs(
        obs=torch.tensor([[1, 2, 3], [1, 2, 3]])+1, reward=np.array([0, 0]), terminal=np.array([False, False]))
    buffer.append_action(action=torch.tensor([[0.2, 0.3], [0.2, 0.3]]))
    buffer.append_obs(
        obs=torch.tensor([[1, 2, 3], [1, 2, 3]])+2, reward=np.array([0, 0]), terminal=np.array([False, False]))
    buffer.append_action(action=torch.tensor([[0.3, 0.4], [0.3, 0.4]]))

    assert (buffer.obs[:, 0, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert torch.isclose(
        buffer.action[:, 0],
        torch.tensor([[.1, .2], [.2, .3], [.3, .4]])
    ).all()
    assert (buffer.obs[:, 1, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert torch.isclose(
        buffer.action[:, 1],
        torch.tensor([[.1, .2], [.2, .3], [.3, .4]])
    ).all()


def test_torch_obs_torch_tuple_action():
    buffer = Buffer(
        num_envs=2,
        max_len=3,
    )

    buffer.append_obs(obs=torch.tensor([[1, 2, 3], [1, 2, 3]]))
    buffer.append_action(
            action=(torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.2]))
    )
    buffer.append_obs(
        obs=torch.tensor([[1, 2, 3], [1, 2, 3]])+1, reward=np.array([0, 0]), terminal=np.array([False, False]))
    buffer.append_action(
            action=(torch.tensor([0.2, 0.3]), torch.tensor([0.2, 0.3]))
    )
    buffer.append_obs(
        obs=torch.tensor([[1, 2, 3], [1, 2, 3]])+2, reward=np.array([0, 0]), terminal=np.array([False, False]))
    buffer.append_action(
            action=(torch.tensor([0.3, 0.4]), torch.tensor([0.3, 0.4]))
    )

    assert (buffer.obs[:, 0, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert torch.isclose(
        buffer.action[0],
        torch.tensor([[.1, .2], [.2, .3], [.3, .4]])
    ).all()
    assert (buffer.obs[:, 1, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert torch.isclose(
        buffer.action[1],
        torch.tensor([[.1, .2], [.2, .3], [.3, .4]])
    ).all()


def test_dict_obs_dict_action():
    buffer = Buffer(
        num_envs=2,
        max_len=3,
    )

    buffer.append_obs(obs={'obs': np.array([[1, 2, 3], [1, 2, 3]])})
    buffer.append_action(action={'a': np.array([0, 0])})
    buffer.append_obs(
        obs={'obs': np.array([[1, 2, 3], [1, 2, 3]])+1}, reward=np.array([0, 0]), terminal=np.array([False, False]))
    buffer.append_action(action={'a': np.array([1, 1])})
    buffer.append_obs(
        obs={'obs': np.array([[1, 2, 3], [1, 2, 3]])+2}, reward=np.array([0, 0]), terminal=np.array([False, False]))

    assert (buffer.obs['obs'][:, 0, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert (buffer.action['a'][:, 0] == torch.tensor([0, 1])).all()
    assert (buffer.obs['obs'][:, 1, :] == torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])).all()
    assert (buffer.action['a'][:, 1] == torch.tensor([0, 1])).all()

# Misc data

def test_misc_list_of_int():
    buffer = Buffer(
        num_envs=1,
        max_len=3,
    )
    buffer.append_obs(obs=np.array([0]), misc=[0])
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc=[1])
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc=[2])

    assert isinstance(buffer.misc, torch.Tensor)
    seq_len, batch_size = buffer.misc.shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.misc == torch.tensor([[0], [1], [2]])).all()


def test_misc_list_of_int_2_envs():
    buffer = Buffer(
        num_envs=2,
        max_len=3,
    )
    buffer.append_obs(obs=np.array([0, 0]), misc=[0, 0])
    buffer.append_action(action=np.array([0, 0]))
    buffer.append_obs(
        obs=np.array([0, 0]), reward=np.array([0, 0]),
        terminal=np.array([False, False]), misc=[1, 1])
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0, 0]), reward=np.array([0, 0]),
        terminal=np.array([False, False]), misc=[2, 2])

    assert isinstance(buffer.misc, torch.Tensor)
    seq_len, batch_size = buffer.misc.shape
    assert seq_len == 3
    assert batch_size == 2
    assert (buffer.misc == torch.tensor([[0], [1], [2]])).all()


def test_misc_list_of_dict():
    buffer = Buffer(
        num_envs=1,
        max_len=3,
    )

    buffer.append_obs(obs=np.array([0]), misc=[{'foo': 0}])
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc=[{'foo': 1}])
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc=[{'foo': 2}])

    assert isinstance(buffer.misc, dict)
    assert 'foo' in buffer.misc
    seq_len, batch_size = buffer.misc['foo'].shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.misc['foo'] == torch.tensor([[0], [1], [2]])).all()


def test_misc_list_of_dict_2_envs():
    buffer = Buffer(
        num_envs=2,
        max_len=3,
    )

    buffer.append_obs(obs=np.array([0]), misc=[{'foo': 0}, {'foo': 1}])
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc=[{'foo': 1}, {'foo': 2}])
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc=[{'foo': 2}, {'foo': 3}])

    assert isinstance(buffer.misc, dict)
    assert 'foo' in buffer.misc
    seq_len, batch_size = buffer.misc['foo'].shape
    assert seq_len == 3
    assert batch_size == 2
    assert (buffer.misc['foo'] == torch.tensor([[0, 1], [1, 2], [2, 3]])).all()


def test_misc_nested_dict():
    buffer = Buffer(
        num_envs=1,
        max_len=3,
    )
    buffer.append_obs(
        obs=np.array([0]),
        misc=[{'foo': 0, 'bar': {'a': .1, 'b': .2}}])
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]),
        reward=np.array([0]),
        terminal=np.array([False]),
        misc=[{'foo': 1, 'bar': {'a': .3, 'b': .4}}])
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]),
        reward=np.array([0]),
        terminal=np.array([False]),
        misc=[{'foo': 2, 'bar': {'a': .5, 'b': .6}}])

    assert isinstance(buffer.misc, dict)

    assert 'foo' in buffer.misc
    seq_len, batch_size = buffer.misc['foo'].shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.misc['foo'] == torch.tensor([[0], [1], [2]])).all()

    assert 'bar' in buffer.misc
    assert isinstance(buffer.misc['bar'], dict)
    assert buffer.misc['bar']['a'].tolist() == [[.1], [.3], [.5]]
    assert buffer.misc['bar']['b'].tolist() == [[.2], [.4], [.6]]


def test_misc_dict_of_tensors():
    """
    If we already have the misc data in batched form, then we would want to pass that data to the `VecHistoryBuffer` in the same form.
    """
    buffer = Buffer(
        num_envs=1,
        max_len=3,
    )

    buffer.append_obs(obs=np.array([0]), misc={'foo': torch.tensor([0])})
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc={'foo': torch.tensor([1])})
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc={'foo': torch.tensor([2])})

    assert isinstance(buffer.misc, dict)
    assert 'foo' in buffer.misc
    seq_len, batch_size = buffer.misc['foo'].shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.misc['foo'] == torch.tensor([[0], [1], [2]])).all()


def test_misc_dict_of_tensors_2_envs():
    """
    If we already have the misc data in batched form, then we would want to pass that data to the `VecHistoryBuffer` in the same form.
    """
    buffer = Buffer(
        num_envs=2,
        max_len=3,
    )

    buffer.append_obs(obs=np.array([0]), misc={'foo': torch.tensor([0, 1])})
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc={'foo': torch.tensor([1, 2])})
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(
        obs=np.array([0]), reward=np.array([0]),
        terminal=np.array([False]), misc={'foo': torch.tensor([2, 3])})

    assert isinstance(buffer.misc, dict)
    assert 'foo' in buffer.misc
    seq_len, batch_size = buffer.misc['foo'].shape
    assert seq_len == 3
    assert batch_size == 2
    assert (buffer.misc['foo'] == torch.tensor([[0, 1], [1, 2], [2, 3]])).all()
