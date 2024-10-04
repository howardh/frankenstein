import pytest
import torch

from frankenstein.algorithms.option_critic import compute_expected_state_value as compute_value
from frankenstein.algorithms.option_critic import compute_expected_state_value_batch as compute_value_batch


@pytest.mark.parametrize("option", [0, 1, 2])
def test_no_termination(option):
    """ If the options do not terminate, then the expected value is the value of the current option """
    state_option_value = torch.tensor([1., 2., 3.])
    val = compute_value(
            state_option_value=state_option_value,
            termination_prob=0.,
            option=option,
            eps=float(torch.rand(1).item()), # Epsilon should not affect the results
            deliberation_cost=float(torch.rand(1).item()), # Deliberation cost should not affect the results
    )
    assert val == state_option_value[option]

@pytest.mark.parametrize("option", [0, 1, 2])
def test_terminates_random_option(option):
    """ If the options terminates and the choice of the next option is random, then the expected value is the mean of the values of all the options """
    state_option_value = torch.tensor([1., 2., 3.])
    val = compute_value(
            state_option_value=state_option_value,
            termination_prob=1.,
            option=option,
            eps=1.,
            deliberation_cost=0.
    )
    assert val == state_option_value.mean()

@pytest.mark.parametrize("option", [0, 1, 2])
def test_terminates_no_randomness(option):
    """ If the options terminates and the choice of the next option is greedy, then the expected value is the values of the best option """
    state_option_value = torch.tensor([1., 2., 3.])
    val = compute_value(
            state_option_value=state_option_value,
            termination_prob=1.,
            option=option,
            eps=0.,
            deliberation_cost=0.
    )
    assert val == 3

@pytest.mark.parametrize("option", [0, 1, 2])
def test_terminates(option):
    """ If the options terminates and the choice of the next option is epsilon-greedy with epsilon=0.5, then the expected value is a linear combination of the values of the best option and the mean of all options. """
    state_option_value = torch.tensor([1., 2., 3.])
    val = compute_value(
            state_option_value=state_option_value,
            termination_prob=1.,
            option=option,
            eps=0.5,
            deliberation_cost=0.
    )
    assert val == 0.5*3 + 0.5*6/3 # (1-eps)*max + eps*mean

def test_batch():
    """ The batched version of the function should be the same as the non-batched version """
    batch_size = 5
    num_options = 3

    state_option_value = torch.rand(batch_size, num_options)
    termination_prob = torch.rand(batch_size)
    option = torch.randint(0, num_options, (batch_size,))
    eps = float(torch.rand(1).item())
    deliberation_cost = float(torch.rand(1).item())

    val_batch = compute_value_batch(
            state_option_value=state_option_value,
            termination_prob=termination_prob,
            option=option,
            eps=eps,
            deliberation_cost=deliberation_cost
    )

    val_no_batch = torch.stack([compute_value(
            state_option_value=state_option_value[i,:],
            termination_prob=float(termination_prob[i].item()),
            option=int(option[i].item()),
            eps=eps,
            deliberation_cost=deliberation_cost
    ) for i in range(batch_size)])

    assert val_batch.shape == val_no_batch.shape
    assert torch.allclose(val_batch, val_no_batch)
