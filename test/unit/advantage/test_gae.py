from pytest import approx
import torch

from frankenstein.advantage.gae import generalized_advantage_estimate as gae
from frankenstein.value.lam import lambda_return_iterative as lambda_return

# gae lambda = 0 => 1-step advantage
# gae lambda = 1 => n-step advantage

# No termination
# Same discount for all steps
# gae lambda = 0


def test_0_steps():
    output = gae(
        state_values=torch.tensor([], dtype=torch.float),
        next_state_values=torch.tensor([], dtype=torch.float),
        rewards=torch.tensor([], dtype=torch.float),
        terminals=torch.tensor([], dtype=torch.float),
        discount=0.9,
        gae_lambda=0,
    )
    assert torch.tensor(output.shape).tolist() == [0]


def test_1_steps():
    output = gae(
        state_values=torch.tensor([5], dtype=torch.float),
        next_state_values=torch.tensor([6], dtype=torch.float),
        rewards=torch.tensor([1], dtype=torch.float),
        terminals=torch.tensor([False], dtype=torch.float),
        discount=0.9,
        gae_lambda=0,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == approx(1+0.9*6 - 5)


def test_3_steps():
    output = gae(
        state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        next_state_values=torch.tensor([6, 7, 8], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        terminals=torch.tensor([False, False, False], dtype=torch.float),
        discount=0.9,
        gae_lambda=0,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[0].item() == approx(1+0.9*6 - 5)
    assert output[1].item() == approx(2+0.9*7 - 6)
    assert output[2].item() == approx(3+0.9*8 - 7)

# With termination
# Same discount for all steps
# gae lambda = 0


def test_termination_at_start():
    output = gae(
        state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        next_state_values=torch.tensor([6, 7, 8], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        terminals=torch.tensor([True, False, False], dtype=torch.float),
        discount=0.9,
        gae_lambda=0,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[0].item() == approx(1 - 5)
    assert output[1].item() == approx(2+0.9*7 - 6)
    assert output[2].item() == approx(3+0.9*8 - 7)


def test_termination_at_end():
    output = gae(
        state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        next_state_values=torch.tensor([6, 7, 8], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        terminals=torch.tensor([False, False, True], dtype=torch.float),
        discount=0.9,
        gae_lambda=0,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[0].item() == approx(1+0.9*6 - 5)
    assert output[1].item() == approx(2+0.9*7 - 6)
    assert output[2].item() == approx(3 - 7)


def test_termination_in_middle():
    output = gae(
        state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        next_state_values=torch.tensor([6, 7, 8], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        terminals=torch.tensor([False, True, False], dtype=torch.float),
        discount=0.9,
        gae_lambda=0,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[0].item() == approx(1+0.9*6 - 5)
    assert output[1].item() == approx(2 - 6)
    assert output[2].item() == approx(3+0.9*8 - 7)


# No termination
# Same discount for all steps
# gae lambda = 1

def test_1_steps_gae_1():
    output = gae(
        state_values=torch.tensor([5], dtype=torch.float),
        next_state_values=torch.tensor([6], dtype=torch.float),
        rewards=torch.tensor([1], dtype=torch.float),
        terminals=torch.tensor([False], dtype=torch.float),
        discount=0.9,
        gae_lambda=1,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == approx(1+0.9*6 - 5)


def test_1_steps_terminal_gae_1():
    output = gae(
        state_values=torch.tensor([5], dtype=torch.float),
        next_state_values=torch.tensor([6], dtype=torch.float),
        rewards=torch.tensor([1], dtype=torch.float),
        terminals=torch.tensor([True], dtype=torch.float),
        discount=0.9,
        gae_lambda=1,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == approx(1 - 5)


def test_2_steps_gae_1():
    output = gae(
        state_values=torch.tensor([5, 6], dtype=torch.float),
        next_state_values=torch.tensor([6, 7], dtype=torch.float),
        rewards=torch.tensor([1, 2], dtype=torch.float),
        terminals=torch.tensor([False, False], dtype=torch.float),
        discount=0.9,
        gae_lambda=1,
    )
    assert torch.tensor(output.shape).tolist() == [2]
    d1 = 2+0.9*7 - 6
    assert output[1].item() == approx(d1)
    d0 = 1+0.9*(2+0.9*7) - 5
    assert output[0].item() == approx(d0)


def test_3_steps_gae_1():
    output = gae(
        state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        next_state_values=torch.tensor([6, 7, 8], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        terminals=torch.tensor([False, False, False], dtype=torch.float),
        discount=0.9,
        gae_lambda=1,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    d2 = 3+0.9*8 - 7
    assert output[2].item() == approx(d2)
    d1 = 2+0.9*(3+0.9*8) - 6
    assert output[1].item() == approx(d1)
    d0 = 1+0.9*(2+0.9*(3+0.9*8)) - 5
    assert output[0].item() == approx(d0)


# With termination
# Same discount for all steps
# gae lambda = 1

def test_termination_at_start_gae_1():
    output = gae(
        state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        next_state_values=torch.tensor([6, 7, 8], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        terminals=torch.tensor([True, False, False], dtype=torch.float),
        discount=0.9,
        gae_lambda=1,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    d2 = 3+0.9*8 - 7
    assert output[2].item() == approx(d2)
    d1 = 2+0.9*(3+0.9*8) - 6
    assert output[1].item() == approx(d1)
    d0 = 1 - 5
    assert output[0].item() == approx(d0)


def test_termination_at_end_gae_1():
    output = gae(
        state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        next_state_values=torch.tensor([6, 7, 8], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        terminals=torch.tensor([False, False, True], dtype=torch.float),
        discount=0.9,
        gae_lambda=1,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    d2 = 3 - 7
    assert output[2].item() == approx(d2)
    d1 = 2+0.9*(3) - 6
    assert output[1].item() == approx(d1)
    d0 = 1+0.9*(2+0.9*(3)) - 5
    assert output[0].item() == approx(d0, 1e-5)


def test_termination_in_middle_gae_1():
    output = gae(
        state_values=torch.tensor([5, 6, 7], dtype=torch.float),
        next_state_values=torch.tensor([6, 7, 8], dtype=torch.float),
        rewards=torch.tensor([1, 2, 3], dtype=torch.float),
        terminals=torch.tensor([False, True, False], dtype=torch.float),
        discount=0.9,
        gae_lambda=1,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    d2 = 3+0.9*8 - 7
    assert output[2].item() == approx(d2)
    d1 = 2 - 6
    assert output[1].item() == approx(d1)
    d0 = 1+0.9*(2) - 5
    assert output[0].item() == approx(d0)


# Compare to values obtained from the lambda returns

def test_gae_lambda_return():
    num_steps = 10
    lam = 0.3
    state_values = torch.rand([num_steps+1])
    rewards = torch.rand([num_steps])
    terminals = (torch.rand([num_steps])*3).floor().bool().logical_not()
    output = gae(
        state_values=state_values[:-1],
        next_state_values=state_values[1:],
        rewards=rewards,
        terminals=terminals,
        discount=0.9,
        gae_lambda=lam,
    )
    output_lambda_return = lambda_return(
        next_state_values=state_values[1:],
        rewards=rewards,
        terminals=terminals,
        discount=0.9,
        lam=lam,
    )
    assert torch.isclose(
        output,
        output_lambda_return-state_values[:-1]
    ).all()


# Make sure the batch version behaves the same as the non-batched version

def test_batch_matches_non_batched():
    num_steps = 5
    batch_size = 10
    state_values = torch.rand([num_steps, batch_size])
    next_state_values = torch.rand([num_steps, batch_size])
    rewards = torch.rand([num_steps, batch_size])
    terminals = (torch.rand([num_steps, batch_size])*2).floor().bool()
    gae_lambda = 0.7
    discount = 0.3
    output_batch = gae(
        state_values=state_values,
        next_state_values=next_state_values,
        rewards=rewards,
        terminals=terminals,
        discount=discount,
        gae_lambda=gae_lambda,
    )
    for i in range(batch_size):
        output = gae(
            state_values=state_values[:, i],
            next_state_values=next_state_values[:, i],
            rewards=rewards[:, i],
            terminals=terminals[:, i],
            discount=discount,
            gae_lambda=gae_lambda,
        )
        assert torch.isclose(output, output_batch[:, i]).all()


def test_batch_matches_non_batched_3_dims():
    num_steps = 5
    batch_size = 10
    batch_size2 = 10
    batch_size3 = 10
    state_values = torch.rand([num_steps, batch_size, batch_size2, batch_size3])
    next_state_values = torch.rand([num_steps, batch_size, batch_size2, batch_size3])
    rewards = torch.rand([num_steps, batch_size, batch_size2, batch_size3])
    terminals = (torch.rand([num_steps, batch_size, batch_size2, batch_size3])*2).floor().bool()
    gae_lambda = 0.7
    discount = 0.3
    output_batch = gae(
        state_values=state_values,
        next_state_values=next_state_values,
        rewards=rewards,
        terminals=terminals,
        discount=discount,
        gae_lambda=gae_lambda,
    )
    for i in range(batch_size):
        for j in range(batch_size2):
            for k in range(batch_size3):
                output = gae(
                    state_values=state_values[:, i, j, k],
                    next_state_values=next_state_values[:, i, j, k],
                    rewards=rewards[:, i, j, k],
                    terminals=terminals[:, i, j, k],
                    discount=discount,
                    gae_lambda=gae_lambda,
                )
                assert torch.isclose(output, output_batch[:, i, j, k]).all()
