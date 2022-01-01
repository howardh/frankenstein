import pytest
import torch
import numpy as np

from frankenstein.buffer.history import HistoryBuffer as Buffer

def test_max_len_0():
    with pytest.raises(Exception):
        Buffer(
                num_envs=1,
                max_len=0,
        )

def test_1_env_default_env_index():
    buffer = Buffer(
            num_envs=1,
            max_len=3,
    )
    buffer.append_obs(obs=np.array([1,2,3]))
    buffer.append_action(action=0)
    buffer.append_obs(obs=np.array([1,2,3])+1, reward=0, terminal=False)
    buffer.append_action(action=1)
    buffer.append_obs(obs=np.array([1,2,3])+2, reward=0, terminal=False)

    seq_len, batch_size, obs_len = buffer.obs.shape
    assert seq_len == 3
    assert batch_size == 1
    assert obs_len == 3
    assert (buffer.obs == torch.tensor([[[1,2,3]],[[2,3,4]],[[3,4,5]]])).all()
    assert (buffer[0].obs == torch.tensor([[1,2,3],[2,3,4],[3,4,5]])).all()

    seq_len, batch_size = buffer.reward.shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.reward == torch.tensor([[0],[0],[0]])).all()
    assert (buffer[0].reward == torch.tensor([0,0,0])).all()

    seq_len, batch_size = buffer.terminal.shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.terminal == torch.tensor([[False],[False],[False]])).all()
    assert (buffer[0].terminal == torch.tensor([False,False,False])).all()

    seq_len, batch_size = buffer.action.shape
    assert seq_len == 2
    assert batch_size == 1
    assert (buffer.action == torch.tensor([[0],[1]])).all()
    assert (buffer[0].action == torch.tensor([0,1])).all()

def test_2_envs_unspecified_index():
    buffer = Buffer(
            num_envs=2,
            max_len=1,
    )
    with pytest.raises(Exception):
        buffer.append_obs(obs=np.array([1,2,3]))

    buffer.append_obs(obs=np.array([1,2,3]), env_index=0)
    with pytest.raises(Exception):
        buffer.append_action(action=0)

def test_episode_termination():
    buffer = Buffer(
            num_envs=1,
            max_len=10,
            default_action=5,
    )
    buffer.append_obs(obs=0)
    buffer.append_action(action=10)
    buffer.append_obs(obs=1,reward=0,terminal=False)
    buffer.append_action(action=11)
    buffer.append_obs(obs=2,reward=0,terminal=True)

    buffer.append_obs(obs=0)
    buffer.append_action(action=10)
    buffer.append_obs(obs=1,reward=0,terminal=False)
    buffer.append_action(action=11)
    buffer.append_obs(obs=2,reward=0,terminal=True)

    buffer[0].action == torch.tensor([10,11,5,10,11])
    buffer[0].terminal == torch.tensor([False,False,True,False,False,True])

def test_exceed_max_len():
    buffer = Buffer(
            num_envs=1,
            max_len=1,
    )
    buffer.append_obs(obs=np.array([1,2,3]))
    buffer.append_action(action=0)
    buffer.append_obs(obs=np.array([1,2,3])+1, reward=0, terminal=False)
    buffer.append_action(action=1)
    buffer.append_obs(obs=np.array([1,2,3])+2, reward=0, terminal=False)

    seq_len, batch_size, obs_len = buffer.obs.shape
    assert seq_len == 2
    assert batch_size == 1
    assert obs_len == 3
    assert (buffer.obs == torch.tensor([[[2,3,4]],[[3,4,5]]])).all()

    seq_len, batch_size = buffer.reward.shape
    assert seq_len == 2
    assert batch_size == 1
    assert (buffer.reward == torch.tensor([[0],[0]])).all()

    seq_len, batch_size = buffer.terminal.shape
    assert seq_len == 2
    assert batch_size == 1
    assert (buffer.terminal == torch.tensor([[False],[False]])).all()

def test_clear():
    buffer = Buffer(
            num_envs=1,
            max_len=10,
            default_action=5,
    )
    buffer.append_obs(obs=0)
    buffer.append_action(action=10)
    buffer.append_obs(obs=1,reward=0,terminal=False)
    buffer.append_action(action=11)
    buffer.append_obs(obs=2,reward=0,terminal=True)

    buffer.append_obs(obs=0)
    buffer.append_action(action=10)
    buffer.append_obs(obs=1,reward=0,terminal=False)
    buffer.append_action(action=11)
    buffer.append_obs(obs=2,reward=0,terminal=True)

    buffer[0].obs_history == [0,1,2,0,1,2]
    buffer[0].reward_history == [0,0,0,0,0,0]
    buffer[0].action_history == [10,11,5,10,11]
    buffer[0].terminal_history == [False,False,True,False,False,True]

    buffer.clear()

    # It should keep the last observation/reward/terminal because it's part of the next transition. The action history should be cleared.
    buffer[0].obs_history == [2]
    buffer[0].reward_history == [0]
    buffer[0].action_history == []
    buffer[0].terminal_history == [True]

def test_dynamic_resizing():
    buffer = Buffer(
            num_envs=0,
            max_len=10,
            default_action=5,
    )
    for i in range(3):
        buffer.append_obs(obs=0,env_index=i)
        buffer.append_action(action=10,env_index=i)
        buffer.append_obs(obs=1,reward=0,terminal=False,env_index=i)
        buffer.append_action(action=11,env_index=i)
        buffer.append_obs(obs=2,reward=0,terminal=True,env_index=i)

        buffer[i].action == torch.tensor([10,11])
        buffer[i].terminal == torch.tensor([False,False,True])

def test_dynamic_resizing_via_slice():
    buffer = Buffer(
            num_envs=0,
            max_len=10,
            default_action=5,
    )
    for i in range(3):
        buffer[i].append_obs(obs=0)
        buffer[i].append_action(action=10)
        buffer[i].append_obs(obs=1,reward=0,terminal=False)
        buffer[i].append_action(action=11)
        buffer[i].append_obs(obs=2,reward=0,terminal=True)

        buffer[i].action == torch.tensor([10,11])
        buffer[i].terminal == torch.tensor([False,False,True])

def test_dynamic_resizing_more_than_1():
    buffer = Buffer(
            num_envs=0,
            max_len=10,
            default_action=5,
    )

    with pytest.raises(Exception):
        buffer.append_obs(obs=0,env_index=1)

    buffer.append_obs(obs=0,env_index=0)

    with pytest.raises(Exception):
        buffer.append_action(action=10,env_index=2)

# Numpy obs, int actions, no misc
def test_numpy_obs_int_action():
    buffer = Buffer(
            num_envs=2,
            max_len=3,
    )
    for i in range(2):
        buffer[i].append_obs(obs=np.array([1,2,3]))
        buffer[i].append_action(action=0)
        buffer[i].append_obs(obs=np.array([1,2,3])+1, reward=0, terminal=False)
        buffer[i].append_action(action=1)
        buffer[i].append_obs(obs=np.array([1,2,3])+2, reward=0, terminal=False)
        buffer[i].append_action(action=2)

        assert (buffer[i].obs == torch.tensor([[1,2,3],[2,3,4],[3,4,5]])).all()
        assert (buffer[i].action == torch.tensor([0,1,2])).all()

def test_numpy_obs_numpy_action():
    buffer = Buffer(
            num_envs=2,
            max_len=3,
    )
    for i in range(2):
        buffer[i].append_obs(obs=np.array([1,2,3]))
        buffer[i].append_action(action=np.array([0.1,0.2],dtype=np.float32))
        buffer[i].append_obs(obs=np.array([1,2,3])+1, reward=0, terminal=False)
        buffer[i].append_action(action=np.array([0.2,0.3],dtype=np.float32))
        buffer[i].append_obs(obs=np.array([1,2,3])+2, reward=0, terminal=False)
        buffer[i].append_action(action=np.array([0.3,0.4],dtype=np.float32))

        assert (buffer[i].obs == torch.tensor([[1,2,3],[2,3,4],[3,4,5]])).all()
        assert torch.isclose(
                buffer[i].action,
                torch.tensor([[.1,.2],[.2,.3],[.3,.4]])
        ).all()

def test_torch_obs_torch_action():
    buffer = Buffer(
            num_envs=2,
            max_len=3,
    )
    for i in range(2):
        buffer[i].append_obs(obs=np.array([1,2,3]))
        buffer[i].append_action(action=torch.tensor([0.1,0.2]))
        buffer[i].append_obs(obs=np.array([1,2,3])+1, reward=0, terminal=False)
        buffer[i].append_action(action=torch.tensor([0.2,0.3]))
        buffer[i].append_obs(obs=np.array([1,2,3])+2, reward=0, terminal=False)
        buffer[i].append_action(action=torch.tensor([0.3,0.4]))

        assert (buffer[i].obs == torch.tensor([[1,2,3],[2,3,4],[3,4,5]])).all()
        assert torch.isclose(
                buffer[i].action,
                torch.tensor([[.1,.2],[.2,.3],[.3,.4]])
        ).all()

def test_tuple_int_action():
    buffer = Buffer(
            num_envs=2,
            max_len=3,
    )
    for i in range(2):
        buffer[i].append_obs(obs=0)
        buffer[i].append_action(action=(0,1))
        buffer[i].append_obs(obs=0, reward=0, terminal=False)
        buffer[i].append_action(action=(2,3))
        buffer[i].append_obs(obs=0, reward=0, terminal=False)
        buffer[i].append_action(action=(4,5))

        assert len(buffer[i].action) == 2
        assert buffer[i].action[0].tolist() == [0,2,4]
        assert buffer[i].action[1].tolist() == [1,3,5]

    assert len(buffer.action) == 2
    seq_len, batch_size = buffer.action[0].shape
    assert seq_len == 3
    assert batch_size == 2
    seq_len, batch_size = buffer.action[1].shape
    assert seq_len == 3
    assert batch_size == 2


# Misc data
def test_misc_int():
    buffer = Buffer(
            num_envs=1,
            max_len=3,
    )
    buffer.append_obs(obs=0, misc=0)
    buffer.append_action(action=0)
    buffer.append_obs(obs=0, reward=0, terminal=False, misc=1)
    buffer.append_action(action=0)
    buffer.append_obs(obs=0, reward=0, terminal=False, misc=2)

    seq_len = buffer[0].misc.shape[0]
    assert seq_len == 3
    assert buffer[0].misc.tolist() == [0,1,2]

    assert isinstance(buffer.misc, torch.Tensor)
    seq_len, batch_size = buffer.misc.shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.misc == torch.tensor([[0],[1],[2]])).all()

def test_misc_dict():
    buffer = Buffer(
            num_envs=1,
            max_len=3,
    )
    buffer.append_obs(obs=0, misc={'foo': 0})
    buffer.append_action(action=0)
    buffer.append_obs(obs=0, reward=0, terminal=False, misc={'foo': 1})
    buffer.append_action(action=0)
    buffer.append_obs(obs=0, reward=0, terminal=False, misc={'foo': 2})

    assert isinstance(buffer[0].misc, dict)
    assert 'foo' in buffer[0].misc
    seq_len = buffer[0].misc['foo'].shape[0]
    assert seq_len == 3
    assert (buffer[0].misc['foo'] == torch.tensor([0,1,2])).all()

    assert isinstance(buffer.misc, dict)
    assert 'foo' in buffer.misc
    seq_len, batch_size = buffer.misc['foo'].shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.misc['foo'] == torch.tensor([[0],[1],[2]])).all()

def test_misc_nested_dict():
    buffer = Buffer(
            num_envs=1,
            max_len=3,
    )
    buffer.append_obs(obs=0, misc={'foo': 0, 'bar': {'a': .1, 'b': .2}})
    buffer.append_action(action=0)
    buffer.append_obs(obs=0, reward=0, terminal=False, misc={'foo': 1, 'bar': {'a': .3, 'b': .4}})
    buffer.append_action(action=0)
    buffer.append_obs(obs=0, reward=0, terminal=False, misc={'foo': 2, 'bar': {'a': .5, 'b': .6}})

    assert isinstance(buffer.misc, dict)

    assert 'foo' in buffer.misc
    seq_len, batch_size = buffer.misc['foo'].shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.misc['foo'] == torch.tensor([[0],[1],[2]])).all()

    assert 'bar' in buffer.misc
    assert isinstance(buffer.misc['bar'], dict)
    assert buffer.misc['bar']['a'].tolist() == [[.1],[.3],[.5]]
    assert buffer.misc['bar']['b'].tolist() == [[.2],[.4],[.6]]

def test_misc_nested_dict_batch_first():
    buffer = Buffer(
            num_envs=1,
            max_len=3,
            batch_first=True,
    )
    buffer.append_obs(obs=0, misc={'foo': 0, 'bar': {'a': .1, 'b': .2}})
    buffer.append_action(action=0)
    buffer.append_obs(obs=0, reward=0, terminal=False, misc={'foo': 1, 'bar': {'a': .3, 'b': .4}})
    buffer.append_action(action=0)
    buffer.append_obs(obs=0, reward=0, terminal=False, misc={'foo': 2, 'bar': {'a': .5, 'b': .6}})

    assert isinstance(buffer.misc, dict)

    assert 'foo' in buffer.misc
    batch_size, seq_len = buffer.misc['foo'].shape
    assert seq_len == 3
    assert batch_size == 1
    assert (buffer.misc['foo'] == torch.tensor([[0,1,2]])).all()

    assert 'bar' in buffer.misc
    assert isinstance(buffer.misc['bar'], dict)
    assert buffer.misc['bar']['a'].tolist() == [[.1,.3,.5]]
    assert buffer.misc['bar']['b'].tolist() == [[.2,.4,.6]]
