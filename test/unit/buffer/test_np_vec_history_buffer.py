import pytest
import torch
import numpy as np

from frankenstein.buffer.vec_history import DataSizes, SerializeFn, default_serialize_np_fn, NumpyBackedVecHistoryBuffer as Buffer
from frankenstein.buffer.vec_history import VecHistoryBuffer as ReferenceBuffer


def make_buffer(data_size: DataSizes, **kwargs):
    obs_dtype = None
    reward_dtype = None
    action_dtype = None
    def get_obs_dtype(x=None):
        nonlocal obs_dtype
        if obs_dtype is None:
            if x is None:
                raise ValueError("obs_dtype is not set and no input provided to infer it.")
            obs_dtype = x.dtype
        return obs_dtype
    def get_reward_dtype(x=None):
        nonlocal reward_dtype
        if reward_dtype is None:
            if x is None:
                raise ValueError("reward_dtype is not set and no input provided to infer it.")
            reward_dtype = x.dtype
        return reward_dtype
    def get_action_dtype(x=None):
        nonlocal action_dtype
        if action_dtype is None:
            if x is None:
                raise ValueError("action_dtype is not set and no input provided to infer it.")
            action_dtype = x.dtype
        return action_dtype

    serialize_fn = SerializeFn(
        obs=default_serialize_np_fn,
        reward=lambda x: x.view(np.uint8).reshape(1,-1,get_reward_dtype(x).itemsize),
        misc=lambda x: np.empty([1,0], dtype=np.uint8),
        action=default_serialize_np_fn,
    )
    deserialize_trajectory_fn = SerializeFn(
        obs=lambda x: torch.from_numpy(x.view(np.float32).copy()).view(x.shape[0], 3),
        reward=lambda x: x.view(np.float32),
        misc=lambda x: np.zeros([x.shape[0],0], dtype=np.uint8),
        action=lambda x: torch.from_numpy(x.view(np.float32).copy()).view(x.shape[0], 4)
    )
    deserialize_transition_fn = SerializeFn(
        obs=lambda x: deserialize_trajectory_fn.obs(x).squeeze(0),
        reward=lambda x: deserialize_trajectory_fn.reward(x).item(),
        misc=lambda x: deserialize_trajectory_fn.misc(x).squeeze(0),
        action=lambda x: deserialize_trajectory_fn.action(x).squeeze(0),
    )

    buffer = Buffer(
        **kwargs,
        data_size=data_size,
        serialize_fn=serialize_fn,
        deserialize_trajectory_fn=deserialize_trajectory_fn,
        deserialize_transition_fn=deserialize_transition_fn,
    )
    return buffer


#@pytest.mark.skip
def test_transitions_match_reference():
    buffer = Buffer(
        max_len=5,
        num_envs=2,
        data_size=DataSizes(3*4, 4, 0, 4*4),
    )
    buffer2 = ReferenceBuffer(
        max_len=5,
        num_envs=2,
    )

    obs = torch.randn(2, 3)
    buffer.append_obs(obs)
    buffer2.append_obs(obs)

    action = torch.randn(2, 4)
    buffer.append_action(action)
    buffer2.append_action(action)

    for i in range(5):
        # Insert random data
        obs = torch.randn(2, 3)
        reward = np.random.randn(2).astype(np.float32)
        term = np.zeros([2], dtype=bool)
        if i == 3:
            term[0] = True
        trunc = np.zeros([2], dtype=bool)
        misc = np.zeros([2,0], dtype=np.uint8)
        buffer.append_obs(obs, reward, term, trunc, misc)
        buffer2.append_obs(obs, reward, term, trunc, misc)

        action = torch.randn(2, 4)
        buffer.append_action(action)
        buffer2.append_action(action)

        # Verify that the data is the same
        assert buffer.num_transitions == buffer2.num_transitions
        for j in range(buffer.num_transitions):
            #if i==3 and j==3:
            #    breakpoint()
            tr = buffer.get_transition(j)
            tr_ref = buffer2.get_transition(j)
            assert tr.obs.shape == tr_ref.obs.shape
            assert tr.action.shape == tr_ref.action.shape
            assert tr.reward == tr_ref.reward
            assert tr.terminated == tr_ref.terminated
            assert tr.truncated == tr_ref.truncated

            assert isinstance(tr.obs, torch.Tensor)
            assert isinstance(tr_ref.obs, torch.Tensor)
            assert torch.allclose(tr.obs, tr_ref.obs)

            assert isinstance(tr.action, torch.Tensor)
            assert isinstance(tr_ref.action, torch.Tensor)
            assert torch.allclose(tr.action, tr_ref.action)

            assert np.allclose(tr.reward, tr_ref.reward)
            assert np.array_equal(tr.terminated, tr_ref.terminated)
            assert np.array_equal(tr.truncated, tr_ref.truncated)


def test_trajectories_match_reference():
    buffer = Buffer(
        max_len=5,
        num_envs=2,
        data_size=DataSizes(3*4, 4, 0, 4*4),
        trajectory_length=3,
    )
    buffer2 = ReferenceBuffer(
        max_len=5,
        num_envs=2,
        trajectory_length=3,
    )

    obs = torch.randn(2, 3)
    buffer.append_obs(obs)
    buffer2.append_obs(obs)

    action = torch.randn(2, 4)
    buffer.append_action(action)
    buffer2.append_action(action)

    for i in range(5):
        # Insert random data
        obs = torch.randn(2, 3)
        reward = np.random.randn(2).astype(np.float32)
        term = np.zeros([2], dtype=bool)
        if i == 3:
            term[0] = True
        trunc = np.zeros([2], dtype=bool)
        misc = np.zeros([2,0], dtype=np.uint8)
        buffer.append_obs(obs, reward, term, trunc, misc)
        buffer2.append_obs(obs, reward, term, trunc, misc)

        action1 = torch.randn(2, 4)
        buffer.append_action(action1)
        buffer2.append_action(action1)

        # Verify that the data is the same
        assert buffer.num_trajectories == buffer2.num_trajectories

        for j in range(buffer.num_trajectories):
            tr = buffer.get_trajectory(j)
            tr_ref = buffer2.get_trajectory(j)
            assert tr.obs.shape == tr_ref.obs.shape
            assert tr.action.shape == tr_ref.action.shape
            assert (tr.reward == tr_ref.reward).all()
            assert (tr.terminated == tr_ref.terminated).all()
            assert (tr.truncated == tr_ref.truncated).all()

            assert isinstance(tr.obs, torch.Tensor)
            assert isinstance(tr_ref.obs, torch.Tensor)
            assert (tr.obs == tr_ref.obs).all()

            assert isinstance(tr.action, torch.Tensor)
            assert isinstance(tr_ref.action, torch.Tensor)
            assert torch.allclose(tr.action, tr_ref.action)

            assert np.allclose(tr.reward, tr_ref.reward)
            assert np.array_equal(tr.terminated, tr_ref.terminated)
            assert np.array_equal(tr.truncated, tr_ref.truncated)


default_data_size = DataSizes(
    obs=8,
    reward=8,
    misc=0,
    action=8,
)


#def test_transition_count_termination_immediate():
#    """ Check that the transition is available if the episode terminates on the first step. """
#    buffer = make_buffer(
#        num_envs=1,
#        max_len=10,
#        data_size=default_data_size,
#    )
#    obs = np.array([0])
#    action = np.array([0])
#    reward = np.array([0.])
#
#    assert len(buffer.transitions) == 0
#
#    buffer.append_obs(obs=obs)
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([True]))
#
#    assert len(buffer.transitions) == 1
#
#
#@pytest.mark.parametrize('num_steps', [1,2,3])
#def test_transition_count_termination_at_end(num_steps):
#    """ Check that the number of transitions is correct when the episode is terminated only at the end. """
#    buffer = make_buffer(
#        num_envs=1,
#        max_len=10,
#        data_size=default_data_size,
#    )
#    obs = np.array([0])
#    action = np.array([0])
#    reward = np.array([0.])
#
#    assert len(buffer.transitions) == 0
#
#    buffer.append_obs(obs=obs)
#    for _ in range(num_steps-1):
#        buffer.append_action(action=action)
#        buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([True]))
#
#    assert len(buffer.transitions) == num_steps
#
#
#def test_transition_count_termination_in_middle():
#    """ Check that the correct transitions are available if the episode terminates in the middle."""
#    buffer = make_buffer(
#        num_envs=1,
#        max_len=10,
#        data_size=default_data_size,
#    )
#    obs = np.array([0])
#    action = np.array([0])
#    reward = np.array([0.])
#
#    assert len(buffer.transitions) == 0
#
#    buffer.append_obs(obs=obs)
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([True]))
#
#    assert len(buffer.transitions) == 1
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
#
#    assert len(buffer.transitions) == 1
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
#
#    assert len(buffer.transitions) == 2
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
#
#    assert len(buffer.transitions) == 3
#
#
#def test_transition_count_overflow():
#    """ Check that the transition count is correct when the buffer overflows and loops around. """
#    buffer = make_buffer(
#        num_envs=1,
#        max_len=3,
#        data_size=default_data_size,
#    )
#
#    obs = np.array([0])
#    action = np.array([0])
#    reward = np.array([0.])
#
#    assert len(buffer.transitions) == 0
#
#    buffer.append_obs(obs=obs)
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
#
#    assert len(buffer.transitions) == 1 # A A
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
#
#    assert len(buffer.transitions) == 2 # A A A
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([True]))
#
#    assert len(buffer.transitions) == 2 # A A A
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
#
#    assert len(buffer.transitions) == 1 # A A B
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
#
#    assert len(buffer.transitions) == 1 # A B B
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
#
#    assert len(buffer.transitions) == 2 # B B B
#
#
#def test_transition_count_vec():
#    """ Check that the transition count is correct if only one environment terminates. """
#    buffer = make_buffer(
#        num_envs=3,
#        max_len=10,
#        data_size=default_data_size,
#    )
#
#    obs = np.array([0,0,0])
#    action = np.array([0,0,0])
#    reward = np.array([0.,0.,0.])
#
#    assert len(buffer.transitions) == 0
#
#    buffer.append_obs(obs=obs)
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))
#
#    assert len(buffer.transitions) == 3
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))
#
#    assert len(buffer.transitions) == 3+3
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,True]))
#
#    assert len(buffer.transitions) == 3+3+3
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))
#
#    assert len(buffer.transitions) == 3+3+3+2
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))
#
#    assert len(buffer.transitions) == 3+3+3+2+3
#
#
#def test_transition_count_overflow_vec():
#    """ Check that the transition count is correct when the buffer overflows and loops around. """
#    buffer = make_buffer(
#        num_envs=3,
#        max_len=3,
#        data_size=default_data_size,
#    )
#
#    obs = np.array([0,0,0])
#    action = np.array([0,0,0])
#    reward = np.array([0.,0.,0.])
#
#    assert len(buffer.transitions) == 0
#
#    buffer.append_obs(obs=obs)
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))
#
#    assert len(buffer.transitions) == 3
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))
#
#    assert len(buffer.transitions) == 3+3
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,True]))
#
#    assert len(buffer.transitions) == 3+3
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))
#
#    assert len(buffer.transitions) == 3+2
#
#    buffer.append_action(action=action)
#    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))
#
#    assert len(buffer.transitions) == 2+3
