from typing import Sequence

import pytest
from pytest import approx
import torch
from torch import multiprocessing as mp
import numpy as np

from frankenstein.buffer.shared import SharedBuffer, transpose, copy_tensor


@pytest.mark.timeout(1)
def test_1Dbox_obs_disc_action_batchsize_1():
    batch_size = 1
    rollout_length = 3
    buffer = SharedBuffer(
            num_buffers = 3,
            rollout_length = rollout_length,
            observation_space = {
                'type': 'box',
                'shape': (3,),
                'dtype': torch.float32,
            },
            action_space = {
                'type': 'discrete',
            },
            misc_space = None,
    )
    for i in range(4):
        buffer.append_obs(
                obs = torch.ones((3,), dtype=torch.float32)*i,
                reward = 0.1*i,
                terminal = False,
                misc = {},
        )
        buffer.append_action(action = i)

    batch = buffer.get_batch(batch_size=batch_size)
    assert isinstance(batch.obs, torch.Tensor)
    assert batch.obs.shape == (rollout_length+1, batch_size, 3)
    assert (batch.obs[0,0,...] == 0).all()
    assert (batch.obs[1,0,...] == 1).all()
    assert (batch.obs[2,0,...] == 2).all()
    assert (batch.obs[3,0,...] == 3).all()

    assert isinstance(batch.action, torch.Tensor)
    assert batch.action.shape == (rollout_length+1, batch_size, 1)
    assert batch.action[0,0].item() == 0
    assert batch.action[1,0].item() == 1
    assert batch.action[2,0].item() == 2
    assert batch.action[3,0].item() == 3

    assert isinstance(batch.reward, torch.Tensor)
    assert batch.reward.shape == (rollout_length+1, batch_size, 1)
    assert batch.reward[0,0].item() == approx(0.0)
    assert batch.reward[1,0].item() == approx(0.1)
    assert batch.reward[2,0].item() == approx(0.2)
    assert batch.reward[3,0].item() == approx(0.3)


@pytest.mark.timeout(1)
def test_1Dbox_obs_disc_action_batchsize_2():
    batch_size = 2
    rollout_length = 3
    buffer = SharedBuffer(
            num_buffers = 3,
            rollout_length = rollout_length,
            observation_space = {
                'type': 'box',
                'shape': (3,),
                'dtype': torch.float32,
            },
            action_space = {
                'type': 'discrete',
            },
            misc_space = None,
    )
    for i in range(4+3):
        buffer.append_obs(
                obs = torch.ones((3,), dtype=torch.float32)*i,
                reward = 0.1*i,
                terminal = False,
                misc = {},
        )
        buffer.append_action(action = i)

    batch = buffer.get_batch(batch_size=batch_size)
    assert isinstance(batch.obs, torch.Tensor)
    assert batch.obs.shape == (rollout_length+1, batch_size, 3)
    assert (batch.obs[0,0,...] == 0).all()
    assert (batch.obs[1,0,...] == 1).all()
    assert (batch.obs[2,0,...] == 2).all()
    assert (batch.obs[3,0,...] == 3).all()
    assert (batch.obs[0,1,...] == 3).all()
    assert (batch.obs[1,1,...] == 4).all()
    assert (batch.obs[2,1,...] == 5).all()
    assert (batch.obs[3,1,...] == 6).all()

    assert isinstance(batch.action, torch.Tensor)
    assert batch.action.shape == (rollout_length+1, batch_size, 1)
    assert batch.action[0,0].item() == 0
    assert batch.action[1,0].item() == 1
    assert batch.action[2,0].item() == 2
    assert batch.action[3,0].item() == 3
    assert batch.action[0,1].item() == 3
    assert batch.action[1,1].item() == 4
    assert batch.action[2,1].item() == 5
    assert batch.action[3,1].item() == 6

    assert isinstance(batch.reward, torch.Tensor)
    assert batch.reward.shape == (rollout_length+1, batch_size, 1)
    assert batch.reward[0,0].item() == approx(0.0)
    assert batch.reward[1,0].item() == approx(0.1)
    assert batch.reward[2,0].item() == approx(0.2)
    assert batch.reward[3,0].item() == approx(0.3)
    assert batch.reward[0,1].item() == approx(0.3)
    assert batch.reward[1,1].item() == approx(0.4)
    assert batch.reward[2,1].item() == approx(0.5)
    assert batch.reward[3,1].item() == approx(0.6)


@pytest.mark.timeout(1)
def test_3Dbox_obs():
    batch_size = 2
    rollout_length = 10
    buffer = SharedBuffer(
            num_buffers = 3,
            rollout_length = rollout_length,
            observation_space = {
                'type': 'box',
                'shape': (4,84,84),
                'dtype': torch.uint8,
            },
            action_space = {
                'type': 'discrete',
            },
            misc_space = None,
    )
    for i in range(21):
        buffer.append_obs(
                obs = torch.ones((4,84,84), dtype=torch.uint8)*i,
                reward = 0.0,
                terminal = False,
                misc = {},
        )
        buffer.append_action(action = i)

    batch = buffer.get_batch(batch_size=batch_size)
    assert isinstance(batch.obs, torch.Tensor)
    assert batch.obs.shape == (rollout_length+1, batch_size, 4, 84, 84)
    assert (batch.obs[0,0,...] == 0).all()
    assert (batch.obs[1,0,...] == 1).all()
    assert (batch.obs[10,0,...] == 10).all()
    assert (batch.obs[0,1,...] == 10).all()
    assert (batch.obs[1,1,...] == 11).all()


@pytest.mark.timeout(1)
def test_dict_obs():
    batch_size = 2
    rollout_length = 10
    buffer = SharedBuffer(
            num_buffers = 2,
            rollout_length=rollout_length,
            observation_space = {
                'type': 'dict',
                'data': {
                    'obs': {
                        'type': 'box',
                        'shape': (4,84,84),
                        'dtype': torch.uint8,
                    },
                    'reward': {
                        'type': 'box',
                        'shape': (1,),
                        'dtype': torch.float32,
                    },
                    'done': {
                        'type': 'box',
                        'shape': (1,),
                        'dtype': torch.bool,
                    },
                }
            },
            action_space = {
                'type': 'discrete',
            },
            misc_space = {
                'type': 'dict',
                'data': {
                    'episode_step_count': {
                        'type': 'box',
                        'shape': (1,),
                        'dtype': torch.int32,
                    },
                    'episode_return': {
                        'type': 'box',
                        'shape': (1,),
                        'dtype': torch.float32,
                    },
                }
            },
    )
    for i in range(21):
        buffer.append_obs(
                obs = {
                    'obs': torch.ones((4,84,84), dtype=torch.uint8)*i,
                    'reward': torch.tensor(i/10, dtype=torch.float32),
                    'done': torch.tensor(False, dtype=torch.bool),
                },
                reward = 0.0,
                terminal = False,
                misc = {
                    'episode_step_count': torch.tensor(i, dtype=torch.int32),
                    'episode_return': torch.tensor(0.0, dtype=torch.float32),
                },
        )
        buffer.append_action(action = i)

    batch = buffer.get_batch(batch_size=batch_size)

    assert isinstance(batch.obs, dict)
    assert isinstance(batch.misc, dict)

    assert isinstance(batch.obs['obs'], torch.Tensor)
    assert batch.obs['obs'].shape == (rollout_length+1, batch_size, 4, 84, 84)
    assert (batch.obs['obs'][0,0,...] == 0).all()
    assert (batch.obs['obs'][10,0,...] == 10).all()
    assert (batch.obs['obs'][0,1,...] == 10).all()
    assert (batch.obs['obs'][10,1,...] == 20).all()

    assert isinstance(batch.obs['reward'], torch.Tensor)
    assert batch.obs['reward'].shape == (rollout_length+1, batch_size, 1)
    assert (batch.obs['reward'][0,0].item() == 0.0)
    assert (batch.obs['reward'][10,0].item() == 1.0)
    assert (batch.obs['reward'][0,1].item() == 1.0)
    assert (batch.obs['reward'][10,1].item() == 2.0)

    assert isinstance(batch.misc['episode_step_count'], torch.Tensor)
    assert batch.misc['episode_step_count'].shape == (rollout_length+1, batch_size, 1)
    assert (batch.misc['episode_step_count'][0,0].item() == 0)
    assert (batch.misc['episode_step_count'][10,0].item() == 10)
    assert (batch.misc['episode_step_count'][0,1].item() == 10)
    assert (batch.misc['episode_step_count'][10,1].item() == 20)


@pytest.mark.timeout(1)
def test_tuple_obs():
    batch_size = 2
    rollout_length = 10
    buffer = SharedBuffer(
            num_buffers = 2,
            rollout_length=rollout_length,
            observation_space = {
                'type': 'tuple',
                'data': [
                    {
                        'type': 'box',
                        'shape': (4,84,84),
                        'dtype': torch.uint8,
                    },{
                        'type': 'box',
                        'shape': (1,),
                        'dtype': torch.float32,
                    },{
                        'type': 'box',
                        'shape': (1,),
                        'dtype': torch.bool,
                    },
                ]
            },
            action_space = {
                'type': 'discrete',
            },
            misc_space = None,
    )
    for i in range(21):
        buffer.append_obs(
                obs = (
                    torch.ones((4,84,84), dtype=torch.uint8)*i,
                    torch.tensor(i/10, dtype=torch.float32),
                    torch.tensor(False, dtype=torch.bool),
                ),
                reward = 0.0,
                terminal = False,
                misc = None,
        )
        buffer.append_action(action = i)

    batch = buffer.get_batch(batch_size=batch_size)

    assert isinstance(batch.obs, Sequence)
    assert isinstance(batch.obs[0], torch.Tensor)
    assert batch.obs[0].shape == (rollout_length+1, batch_size, 4, 84, 84)
    assert (batch.obs[0][0,0,...] == 0).all()
    assert (batch.obs[0][10,0,...] == 10).all()
    assert (batch.obs[0][0,1,...] == 10).all()
    assert (batch.obs[0][10,1,...] == 20).all()


@pytest.mark.timeout(1)
def test_terminal_in_middle():
    batch_size = 2
    rollout_length = 3
    buffer = SharedBuffer(
            num_buffers = 3,
            rollout_length = rollout_length,
            observation_space = {
                'type': 'box',
                'shape': (3,),
                'dtype': torch.float32,
            },
            action_space = {
                'type': 'discrete',
            },
            misc_space = None,
    )
    for i in range(4+3):
        buffer.append_obs(
                obs = torch.ones((3,), dtype=torch.float32)*i,
                reward = 0.1*i,
                terminal = i==2,
                misc = {},
        )
        if i != 2:
            buffer.append_action(action = i)

    batch = buffer.get_batch(batch_size=batch_size)
    assert isinstance(batch.obs, torch.Tensor)
    assert batch.obs.shape == (rollout_length+1, batch_size, 3)
    assert (batch.obs[0,0,...] == 0).all()
    assert (batch.obs[1,0,...] == 1).all()
    assert (batch.obs[2,0,...] == 2).all()
    assert (batch.obs[3,0,...] == 3).all()
    assert (batch.obs[0,1,...] == 3).all()
    assert (batch.obs[1,1,...] == 4).all()
    assert (batch.obs[2,1,...] == 5).all()
    assert (batch.obs[3,1,...] == 6).all()

    assert isinstance(batch.action, torch.Tensor)
    assert batch.action.shape == (rollout_length+1, batch_size, 1)
    assert batch.action[0,0].item() == 0
    assert batch.action[1,0].item() == 1
    #assert batch.action[2,0].item() == 2 # This is a terminal step, so no action is recorded
    assert batch.action[3,0].item() == 3
    assert batch.action[0,1].item() == 3
    assert batch.action[1,1].item() == 4
    assert batch.action[2,1].item() == 5
    assert batch.action[3,1].item() == 6

    assert isinstance(batch.reward, torch.Tensor)
    assert batch.reward.shape == (rollout_length+1, batch_size, 1)
    assert batch.reward[0,0].item() == approx(0.0)
    assert batch.reward[1,0].item() == approx(0.1)
    assert batch.reward[2,0].item() == approx(0.2)
    assert batch.reward[3,0].item() == approx(0.3)
    assert batch.reward[0,1].item() == approx(0.3)
    assert batch.reward[1,1].item() == approx(0.4)
    assert batch.reward[2,1].item() == approx(0.5)
    assert batch.reward[3,1].item() == approx(0.6)


@pytest.mark.timeout(1)
def test_terminal_at_end():
    batch_size = 2
    rollout_length = 3
    buffer = SharedBuffer(
            num_buffers = 3,
            rollout_length = rollout_length,
            observation_space = {
                'type': 'box',
                'shape': (3,),
                'dtype': torch.float32,
            },
            action_space = {
                'type': 'discrete',
            },
            misc_space = None,
    )
    for i in range(4+4):
        buffer.append_obs(
                obs = torch.ones((3,), dtype=torch.float32)*i,
                reward = 0.1*i,
                terminal = i==3,
                misc = {},
        )
        if i != 3:
            buffer.append_action(action = i)

    batch = buffer.get_batch(batch_size=batch_size)
    assert isinstance(batch.obs, torch.Tensor)
    assert batch.obs.shape == (rollout_length+1, batch_size, 3)
    assert (batch.obs[0,0,...] == 0).all()
    assert (batch.obs[1,0,...] == 1).all()
    assert (batch.obs[2,0,...] == 2).all()
    assert (batch.obs[3,0,...] == 3).all()
    assert (batch.obs[0,1,...] == 4).all()
    assert (batch.obs[1,1,...] == 5).all()
    assert (batch.obs[2,1,...] == 6).all()
    assert (batch.obs[3,1,...] == 7).all()

    assert isinstance(batch.action, torch.Tensor)
    assert batch.action.shape == (rollout_length+1, batch_size, 1)
    assert batch.action[0,0].item() == 0
    assert batch.action[1,0].item() == 1
    assert batch.action[2,0].item() == 2
    #assert batch.action[3,0].item() == 3 # This is a terminal step, so no action is recorded
    assert batch.action[0,1].item() == 4
    assert batch.action[1,1].item() == 5
    assert batch.action[2,1].item() == 6
    assert batch.action[3,1].item() == 7

    assert isinstance(batch.reward, torch.Tensor)
    assert batch.reward.shape == (rollout_length+1, batch_size, 1)
    assert batch.reward[0,0].item() == approx(0.0)
    assert batch.reward[1,0].item() == approx(0.1)
    assert batch.reward[2,0].item() == approx(0.2)
    assert batch.reward[3,0].item() == approx(0.3)
    assert batch.reward[0,1].item() == approx(0.4)
    assert batch.reward[1,1].item() == approx(0.5)
    assert batch.reward[2,1].item() == approx(0.6)
    assert batch.reward[3,1].item() == approx(0.7)


@pytest.mark.timeout(1)
def test_release():
    batch_size = 2
    rollout_length = 3
    buffer = SharedBuffer(
            num_buffers = 2,
            rollout_length = rollout_length,
            observation_space = {
                'type': 'box',
                'shape': (3,),
                'dtype': torch.float32,
            },
            action_space = {
                'type': 'discrete',
            },
            misc_space = None,
    )
    def generate_data():
        for i in range(4+3):
            buffer.append_obs(
                    obs = torch.ones((3,), dtype=torch.float32)*i,
                    reward = 0.1*i,
                    terminal = False,
                    misc = {},
            )
            buffer.append_action(action = i)
    for _ in range(2):
        proc = mp.Process(target=generate_data)
        proc.start()

    batch = buffer.get_batch(batch_size=batch_size)
    batch.release()
    batch = buffer.get_batch(batch_size=batch_size)


# Test util functions

def test_copy_tensor():
    src = torch.rand((3,4))
    dest = torch.zeros((3,4))
    assert (src != dest).any()
    copy_tensor(src=src, dest=dest)
    assert (src == dest).all()
    assert (dest != 0).all()


def test_copy_np_src():
    src = np.array([[1,2,3],[4,5,6]])
    dest = torch.zeros((2,3))

    copy_tensor(src=src, dest=dest)

    assert dest.tolist() == [[1,2,3],[4,5,6]]


def test_copy_dict():
    src = {
        'a': torch.tensor([1,2,3]),
        'b': torch.tensor([4,5,6]),
    }
    dest = {
        'a': torch.tensor([0,0,0]),
        'b': torch.tensor([0,0,0]),
    }
    copy_tensor(src=src, dest=dest)

    assert dest['a'].tolist() == [1,2,3]
    assert dest['b'].tolist() == [4,5,6]


def test_copy_dict_incomplete():
    src = {
        'a': torch.tensor([1,2,3]),
    }
    dest = {
        'a': torch.tensor([0,0,0]),
        'b': torch.tensor([0,0,0]),
    }
    copy_tensor(src=src, dest=dest)

    assert dest['a'].tolist() == [1,2,3]
    assert dest['b'].tolist() == [0,0,0]


def test_copy_invalid_src():
    src = {
        'a': None,
    }
    dest = {
        'a': torch.tensor([0,0,0]),
        'b': torch.tensor([0,0,0]),
    }
    with pytest.raises(TypeError):
        copy_tensor(src=src, dest=dest)


def test_transpose_tensor():
    x = torch.tensor([[1,2,3],[4,5,6]])
    output = transpose(x, 0, 1)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3,2)


def test_transpose_dict():
    x = {
        'a': torch.tensor([[1,2,3],[4,5,6]]),
        'b': torch.tensor([[7,8]]),
    }
    output = transpose(x, 0, 1)

    assert isinstance(output, dict)
    assert output['a'].shape == (3,2)
    assert output['b'].shape == (2,1)
    assert output['a'].tolist() == [[1,4],[2,5],[3,6]]
    assert output['b'].tolist() == [[7],[8]]


def test_transpose_tuple():
    x = (
        torch.tensor([[1,2,3],[4,5,6]]),
        torch.tensor([[7,8]]),
    )
    output = transpose(x, 0, 1)

    assert isinstance(output, tuple)
    assert output[0].shape == (3,2)
    assert output[1].shape == (2,1)
    assert output[0].tolist() == [[1,4],[2,5],[3,6]]
    assert output[1].tolist() == [[7],[8]]


def test_transpose_list():
    x = [
        torch.tensor([[1,2,3],[4,5,6]]),
        torch.tensor([[7,8]]),
    ]
    output = transpose(x, 0, 1)

    assert isinstance(output, list)
    assert output[0].shape == (3,2)
    assert output[1].shape == (2,1)
    assert output[0].tolist() == [[1,4],[2,5],[3,6]]
    assert output[1].tolist() == [[7],[8]]
