import pytest
import torch
import numpy as np

from frankenstein.buffer.vec_history import VecHistoryBuffer as Buffer


def test_max_len_0():
    """ It should not allow creation of a buffer with 0 capacity. """
    with pytest.raises(Exception):
        Buffer(
            num_envs=1,
            max_len=0,
        )


def test_num_envs_0():
    """ It should not allow creation of a buffer with no environments. """
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
        obs=np.array([[1, 2, 3]])+1, reward=np.array([0]), terminated=np.array([False]))
    buffer.append_action(action=np.array([1]))
    buffer.append_obs(
        obs=np.array([[1, 2, 3]])+2, reward=np.array([0]), terminated=np.array([False]))

    assert len(buffer.transitions) == 2

    transition = buffer.transitions[0]
    assert (transition.obs == torch.tensor([[1, 2, 3]])).all()
    assert (transition.next_obs == torch.tensor([[2, 3, 4]])).all()

    transition = buffer.transitions[1]
    assert (transition.obs == torch.tensor([[2, 3, 4]])).all()
    assert (transition.next_obs == torch.tensor([[3, 4, 5]])).all()


# Test data validation

def test_obs_too_long():
    """ If the length of the inputs don't match up with the `num_envs` parameter, then it should raise an error. """
    buffer = Buffer(
        num_envs=4,
        max_len=10,
    )
    obs = np.array([0]*5)
    action = np.array([0]*4)
    reward = np.array([0.]*4)

    with pytest.raises(Exception):
        buffer.append_obs(obs=obs)


def test_action_too_long():
    """ If the length of the inputs don't match up with the `num_envs` parameter, then it should raise an error. """
    buffer = Buffer(
        num_envs=4,
        max_len=10,
    )
    obs = np.array([0]*4)
    action = np.array([0]*5)

    buffer.append_obs(obs=obs)
    with pytest.raises(Exception):
        buffer.append_action(action=action)


def test_reward_too_long():
    """ If the length of the inputs don't match up with the `num_envs` parameter, then it should raise an error. """
    buffer = Buffer(
        num_envs=4,
        max_len=10,
    )
    obs = np.array([0]*4)
    action = np.array([0]*4)
    reward = np.array([0.]*5)
    terminated = np.array([False]*4)
    truncated = np.array([False]*4)

    buffer.append_obs(obs=obs)
    buffer.append_action(action=action)
    with pytest.raises(Exception):
        buffer.append_obs(obs=obs, reward=reward, terminated=terminated, truncated=truncated)


def test_terminated_too_long():
    """ If the length of the inputs don't match up with the `num_envs` parameter, then it should raise an error. """
    buffer = Buffer(
        num_envs=4,
        max_len=10,
    )
    obs = np.array([0]*4)
    action = np.array([0]*4)
    reward = np.array([0.]*4)
    terminated = np.array([False]*5)
    truncated = np.array([False]*4)

    buffer.append_obs(obs=obs)
    buffer.append_action(action=action)
    with pytest.raises(Exception):
        buffer.append_obs(obs=obs, reward=reward, terminated=terminated, truncated=truncated)


def test_truncated_too_long():
    """ If the length of the inputs don't match up with the `num_envs` parameter, then it should raise an error. """
    buffer = Buffer(
        num_envs=4,
        max_len=10,
    )
    obs = np.array([0]*4)
    action = np.array([0]*4)
    reward = np.array([0.]*4)
    terminated = np.array([False]*4)
    truncated = np.array([False]*5)

    buffer.append_obs(obs=obs)
    buffer.append_action(action=action)
    with pytest.raises(Exception):
        buffer.append_obs(obs=obs, reward=reward, terminated=terminated, truncated=truncated)


def test_dict_obs_validation():
    """ If a dict is used as an observation, it should check that all elements of the dict match the number of environments in the first dimension. """
    buffer = Buffer(
        num_envs=4,
        max_len=10,
    )
    obs = {'a': np.array([0]*4), 'b': np.array([0]*4)} # Each element is size 4, so this should not error.
    buffer.append_obs(obs=obs)


# Test transition count


def test_transition_count_termination_immediate():
    """ Check that the transition is available if the episode terminates on the first step. """
    buffer = Buffer(
        num_envs=1,
        max_len=10,
    )
    obs = np.array([0])
    action = np.array([0])
    reward = np.array([0.])

    assert len(buffer.transitions) == 0

    buffer.append_obs(obs=obs)
    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([True]))

    assert len(buffer.transitions) == 1


@pytest.mark.parametrize('num_steps', [1,2,3])
def test_transition_count_termination_at_end(num_steps):
    """ Check that the number of transitions is correct when the episode is terminated only at the end. """
    buffer = Buffer(
        num_envs=1,
        max_len=10,
    )
    obs = np.array([0])
    action = np.array([0])
    reward = np.array([0.])

    assert len(buffer.transitions) == 0

    buffer.append_obs(obs=obs)
    for _ in range(num_steps-1):
        buffer.append_action(action=action)
        buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))
    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([True]))

    assert len(buffer.transitions) == num_steps


def test_transition_count_termination_in_middle():
    """ Check that the correct transitions are available if the episode terminates in the middle."""
    buffer = Buffer(
        num_envs=1,
        max_len=10,
    )
    obs = np.array([0])
    action = np.array([0])
    reward = np.array([0.])

    assert len(buffer.transitions) == 0

    buffer.append_obs(obs=obs)
    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([True]))

    assert len(buffer.transitions) == 1

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))

    assert len(buffer.transitions) == 1

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))

    assert len(buffer.transitions) == 2

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))

    assert len(buffer.transitions) == 3


def test_transition_count_overflow():
    """ Check that the transition count is correct when the buffer overflows and loops around. """
    buffer = Buffer(
        num_envs=1,
        max_len=3,
    )

    obs = np.array([0])
    action = np.array([0])
    reward = np.array([0.])

    assert len(buffer.transitions) == 0

    buffer.append_obs(obs=obs)
    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))

    assert len(buffer.transitions) == 1 # A A

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))

    assert len(buffer.transitions) == 2 # A A A

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([True]))

    assert len(buffer.transitions) == 2 # A A A

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))

    assert len(buffer.transitions) == 1 # A A B

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))

    assert len(buffer.transitions) == 1 # A B B

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False]))

    assert len(buffer.transitions) == 2 # B B B


def test_transition_count_vec():
    """ Check that the transition count is correct if only one environment terminates. """
    buffer = Buffer(
        num_envs=3,
        max_len=10,
    )

    obs = np.array([0,0,0])
    action = np.array([0,0,0])
    reward = np.array([0.,0.,0.])

    assert len(buffer.transitions) == 0

    buffer.append_obs(obs=obs)
    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))

    assert len(buffer.transitions) == 3

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))

    assert len(buffer.transitions) == 3+3

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,True]))

    assert len(buffer.transitions) == 3+3+3

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))

    assert len(buffer.transitions) == 3+3+3+2

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))

    assert len(buffer.transitions) == 3+3+3+2+3


def test_transition_count_overflow_vec():
    """ Check that the transition count is correct when the buffer overflows and loops around. """
    buffer = Buffer(
        num_envs=3,
        max_len=3,
    )

    obs = np.array([0,0,0])
    action = np.array([0,0,0])
    reward = np.array([0.,0.,0.])

    assert len(buffer.transitions) == 0

    buffer.append_obs(obs=obs)
    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))

    assert len(buffer.transitions) == 3

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))

    assert len(buffer.transitions) == 3+3

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,True]))

    assert len(buffer.transitions) == 3+3

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))

    assert len(buffer.transitions) == 3+2

    buffer.append_action(action=action)
    buffer.append_obs(obs=obs, reward=reward, terminated=np.array([False,False,False]))

    assert len(buffer.transitions) == 2+3


# Test transitions


def test_transition_out_of_bounds():
    ...


def test_transition_one_transition():
    """ """
    buffer = Buffer(
        num_envs=1,
        max_len=10,
    )

    buffer.append_obs(obs=np.array([[1, 2, 3]])+0)
    buffer.append_action(action=np.array([0]))
    buffer.append_obs(obs=np.array([[1, 2, 3]])+1, reward=np.array([0]), terminated=np.array([False]))

    transition = buffer.transitions[0]
    assert (transition.obs == torch.tensor([[1, 2, 3]])).all()
    assert (transition.next_obs == torch.tensor([[2, 3, 4]])).all()
    assert (transition.reward == torch.tensor([0])).all()

    buffer.append_action(action=np.array([0]))


def test_transition_vec():
    buffer = Buffer(
        num_envs=2,
        max_len=10,
    )
    
    obs = np.array([[1, 1], [2, 2]])

    buffer.append_obs(obs=obs)

    buffer.append_action(action=np.array([0, 0]))
    buffer.append_obs(obs=obs, reward=np.array([0, 0]), terminated=np.array([False, False]))

    buffer.append_action(action=np.array([1, 1]))
    buffer.append_obs(obs=obs, reward=np.array([0, 0]), terminated=np.array([False, False]))

    buffer.append_action(action=np.array([2, 2]))
    buffer.append_obs(obs=obs, reward=np.array([0, 0]), terminated=np.array([False, False]))

    transition = buffer.transitions[0]
    assert (transition.obs == torch.tensor([[1, 1]])).all()
    assert (transition.next_obs == torch.tensor([[1, 1]])).all()
    assert (transition.action == torch.tensor([0])).all()

    transition = buffer.transitions[2]
    assert (transition.obs == torch.tensor([[1, 1]])).all()
    assert (transition.next_obs == torch.tensor([[1, 1]])).all()
    assert (transition.action == torch.tensor([2])).all()

    transition = buffer.transitions[3]
    assert (transition.obs == torch.tensor([[2, 2]])).all()
    assert (transition.next_obs == torch.tensor([[2, 2]])).all()
    assert (transition.action == torch.tensor([0])).all()

    transition = buffer.transitions[5]
    assert (transition.obs == torch.tensor([[2, 2]])).all()
    assert (transition.next_obs == torch.tensor([[2, 2]])).all()
    assert (transition.action == torch.tensor([2])).all()


def test_transition_with_termination_vec():
    buffer = Buffer(
        num_envs=2,
        max_len=10,
    )
    
    obs = np.array([[1, 1], [2, 2]])

    buffer.append_obs(obs=obs)

    buffer.append_action(action=np.array([0, 0]))
    buffer.append_obs(obs=obs, reward=np.array([0, 0]), terminated=np.array([False, False]))

    buffer.append_action(action=np.array([1, 1]))
    buffer.append_obs(obs=obs, reward=np.array([0, 0]), terminated=np.array([True, False]))

    buffer.append_action(action=np.array([-1, 2]))
    buffer.append_obs(obs=obs, reward=np.array([0, 0]), terminated=np.array([False, False]))

    transition = buffer.transitions[0]
    assert (transition.obs == torch.tensor([[1, 1]])).all()
    assert (transition.next_obs == torch.tensor([[1, 1]])).all()
    assert (transition.action == torch.tensor([0])).all()

    transition = buffer.transitions[1]
    assert (transition.obs == torch.tensor([[1, 1]])).all()
    assert (transition.next_obs == torch.tensor([[1, 1]])).all()
    assert (transition.action == torch.tensor([1])).all()

    transition = buffer.transitions[2]
    assert (transition.obs == torch.tensor([[2, 2]])).all()
    assert (transition.next_obs == torch.tensor([[2, 2]])).all()
    assert (transition.action == torch.tensor([0])).all()

    transition = buffer.transitions[4]
    assert (transition.obs == torch.tensor([[2, 2]])).all()
    assert (transition.next_obs == torch.tensor([[2, 2]])).all()
    assert (transition.action == torch.tensor([2])).all()


def test_transition_batch():
    buffer = Buffer(
        num_envs=2,
        max_len=10,
    )
    
    obs = np.array([[1, 1], [2, 2]])

    buffer.append_obs(obs=obs)

    buffer.append_action(action=np.array([0, 0]))
    buffer.append_obs(obs=obs, reward=np.array([0, 0]), terminated=np.array([False, False]))

    buffer.append_action(action=np.array([1, 1]))
    buffer.append_obs(obs=obs, reward=np.array([0, 0]), terminated=np.array([True, False]))

    buffer.append_action(action=np.array([-1, 2]))
    buffer.append_obs(obs=obs, reward=np.array([0, 0]), terminated=np.array([False, False]))

    batch = buffer.transitions.sample_batch(4)
    assert batch.obs.shape == (4, 2)
    assert batch.next_obs.shape == (4, 2)
    assert batch.action.shape == (4,)
    assert batch.reward.shape == (4, 1)
    assert batch.terminated.shape == (4, 1)
    assert batch.truncated.shape == (4, 1)


@pytest.mark.skip
def test_get_transition_performance():
    import gymnasium
    import timeit

    buffer = Buffer(
        num_envs=1,
        max_len=5_000,
    )

    env = gymnasium.make_vec(
        'HalfCheetah-v4', num_envs=1,
    )
    obs, _ = env.reset()
    buffer.append_obs(obs)
    for _ in range(5000):
        action = env.action_space.sample()
        buffer.append_action(action)
        obs, reward, term, trunc, _ = env.step(action)
        buffer.append_obs(obs, reward, term, trunc)

    fn = lambda: buffer.transitions.sample_batch(32)
    t = timeit.timeit(fn, number=10)
    print(t)

    assert False
