from pytest import approx
import torch

from frankenstein.value.monte_carlo import monte_carlo_return_iterative as state_value
from frankenstein.value.monte_carlo import monte_carlo_return_iterative_batch as state_value_batch


# No termination
# Same discount for all steps


def test_0_steps():
    output = state_value(
        next_state_values=torch.tensor([], dtype=torch.float),
        rewards=torch.tensor([], dtype=torch.float),
        truncated=torch.tensor([]),
        terminated=torch.tensor([]),
        discounts=torch.tensor([], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [0]


def test_1_steps():
    output = state_value(
        next_state_values=torch.tensor([5], dtype=torch.float),
        rewards=torch.tensor([1], dtype=torch.float),
        truncated=torch.tensor([False]),
        terminated=torch.tensor([False]),
        discounts=torch.tensor([0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == approx(1+0.9*5)


def test_3_steps():
    output = state_value(
        next_state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        truncated=torch.tensor([False, False, False]),
        terminated=torch.tensor([False, False, False]),
        discounts=torch.tensor([0.9, 0.9, 0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[2].item() == approx(3+0.9*7)
    assert output[1].item() == approx(2+0.9*(3+0.9*7))
    assert output[0].item() == approx(1+0.9*(2+0.9*(3+0.9*7)))


# With termination
# Same discount for all steps


def test_termination_at_start():
    output = state_value(
        next_state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        truncated=torch.tensor([False, False, False], dtype=torch.float),  # Should work with float or bool
        terminated=torch.tensor([True, False, False], dtype=torch.float),  # Should work with float or bool
        discounts=torch.tensor([0.9, 0.9, 0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[2].item() == approx(3+0.9*7)
    assert output[1].item() == approx(2+0.9*(3+0.9*7))
    assert output[0].item() == approx(1)


def test_termination_at_end():
    output = state_value(
        next_state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        truncated=torch.tensor([False, False, False], dtype=torch.float),  # Should work with float or bool
        terminated=torch.tensor([False, False, True], dtype=torch.float),  # Should work with float or bool
        discounts=torch.tensor([0.9, 0.9, 0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[2].item() == approx(3)
    assert output[1].item() == approx(2+0.9*(3))
    assert output[0].item() == approx(1+0.9*(2+0.9*(3)))


def test_termination_in_middle():
    output = state_value(
        next_state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        truncated=torch.tensor([False, False, False], dtype=torch.bool),  # Should work with float or bool
        terminated=torch.tensor([False, True, False], dtype=torch.bool),  # Should work with float or bool
        discounts=torch.tensor([0.9, 0.9, 0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[2].item() == approx(3+0.9*7)
    assert output[1].item() == approx(2)
    assert output[0].item() == approx(1+0.9*2)


# Truncated without termination
# Same discount for all steps


def test_truncation_at_start():
    output = state_value(
        next_state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        truncated=torch.tensor([True, False, False], dtype=torch.float),  # Should work with float or bool
        terminated=torch.tensor([False, False, False], dtype=torch.float),  # Should work with float or bool
        discounts=torch.tensor([0.9, 0.9, 0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[2].item() == approx(3+0.9*7)
    assert output[1].item() == approx(2+0.9*(3+0.9*7))
    assert output[0].item() == approx(1+0.9*5)


def test_truncation_at_end():
    output = state_value(
        next_state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        truncated=torch.tensor([False, False, True], dtype=torch.float),  # Should work with float or bool
        terminated=torch.tensor([False, False, False], dtype=torch.float),  # Should work with float or bool
        discounts=torch.tensor([0.9, 0.9, 0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[2].item() == approx(3+0.9*7)
    assert output[1].item() == approx(2+0.9*(3+0.9*7))
    assert output[0].item() == approx(1+0.9*(2+0.9*(3+0.9*7)))


def test_truncation_in_middle():
    output = state_value(
        next_state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        truncated=torch.tensor([False, True, False], dtype=torch.bool),  # Should work with float or bool
        terminated=torch.tensor([False, False, False], dtype=torch.bool),  # Should work with float or bool
        discounts=torch.tensor([0.9, 0.9, 0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[2].item() == approx(3+0.9*7)
    assert output[1].item() == approx(2+0.9*6)
    assert output[0].item() == approx(1+0.9*(2+0.9*6))


# Batched


def test_0_batch():
    output = state_value_batch(
        next_state_values=torch.tensor([], dtype=torch.float),
        rewards=torch.tensor([], dtype=torch.float),
        truncated=torch.tensor([]),
        terminated=torch.tensor([]),
        discounts=torch.tensor([], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [0]


def test_0_steps_1_batch():
    output = state_value_batch(
        next_state_values=torch.tensor([[]], dtype=torch.float).t(),
        rewards=torch.tensor([[]], dtype=torch.float).t(),
        truncated=torch.tensor([[]]).t(),
        terminated=torch.tensor([[]]).t(),
        discounts=torch.tensor([[]], dtype=torch.float).t(),
    )
    assert torch.tensor(output.shape).tolist() == [0, 1]


def test_0_steps_2_batch():
    output = state_value_batch(
        next_state_values=torch.tensor([[], []], dtype=torch.float).t(),
        rewards=torch.tensor([[], []], dtype=torch.float).t(),
        truncated=torch.tensor([[], []]).t(),
        terminated=torch.tensor([[], []]).t(),
        discounts=torch.tensor([[], []], dtype=torch.float).t(),
    )
    assert torch.tensor(output.shape).tolist() == [0, 2]


# Make sure the batch version behaves the same as the non-batched version


def test_batch_matches_non_batched():
    num_steps = 5
    batch_size = 10
    state_values = torch.rand([num_steps, batch_size])
    rewards = torch.rand([num_steps, batch_size])
    terminated = (torch.rand([num_steps, batch_size])*2).floor().bool()
    truncated = (torch.rand([num_steps, batch_size])*2).floor().bool()
    discounts = torch.rand([num_steps, batch_size])
    output_batch = state_value_batch(
        next_state_values=state_values,
        rewards=rewards,
        truncated=truncated,
        terminated=terminated,
        discounts=discounts,
    )
    for i in range(batch_size):
        output = state_value(
            next_state_values=state_values[:, i],
            rewards=rewards[:, i],
            truncated=truncated[:, i],
            terminated=terminated[:, i],
            discounts=discounts[:, i],
        )
        assert torch.isclose(output, output_batch[:, i]).all()
