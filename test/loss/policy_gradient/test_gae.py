from pytest import approx
import torch

from frankenstein.loss.policy_gradient import geneeralized_advantage_estimate as gae

# gae lambda = 0 => 1-step advantage
# gae lambda = 1 => n-step advantage

# No termination
# Same discount for all steps
# gae lambda = 0
def test_0_steps():
    output = gae(
            state_values = torch.tensor([], dtype=torch.float),
            next_state_values = torch.tensor([], dtype=torch.float),
            rewards = torch.tensor([], dtype=torch.float),
            terminals = torch.tensor([], dtype=torch.float),
            discounts = torch.tensor([], dtype=torch.float),
            gae_lambda = 0,
    )
    assert torch.tensor(output.shape).tolist() == [0]

def test_1_steps():
    output = gae(
            state_values = torch.tensor([5], dtype=torch.float),
            next_state_values = torch.tensor([6], dtype=torch.float),
            rewards = torch.tensor([1], dtype=torch.float),
            terminals = torch.tensor([False], dtype=torch.float),
            discounts = torch.tensor([0.9], dtype=torch.float),
            gae_lambda = 0,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == approx(1+0.9*6 - 5)

def test_3_steps():
    output = gae(
            state_values = torch.tensor([5,6,7], dtype=torch.float),
            next_state_values = torch.tensor([6,7,8], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            gae_lambda = 0,
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
            state_values = torch.tensor([5,6,7], dtype=torch.float),
            next_state_values = torch.tensor([6,7,8], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([True,False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            gae_lambda = 0,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[0].item() == approx(1 - 5)
    assert output[1].item() == approx(2+0.9*7 - 6)
    assert output[2].item() == approx(3+0.9*8 - 7)

def test_termination_at_end():
    output = gae(
            state_values = torch.tensor([5,6,7], dtype=torch.float),
            next_state_values = torch.tensor([6,7,8], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,False,True], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            gae_lambda = 0,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[0].item() == approx(1+0.9*6 - 5)
    assert output[1].item() == approx(2+0.9*7 - 6)
    assert output[2].item() == approx(3 - 7)

def test_termination_in_middle():
    output = gae(
            state_values = torch.tensor([5,6,7], dtype=torch.float),
            next_state_values = torch.tensor([6,7,8], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,True,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            gae_lambda = 0,
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
            state_values = torch.tensor([5], dtype=torch.float),
            next_state_values = torch.tensor([6], dtype=torch.float),
            rewards = torch.tensor([1], dtype=torch.float),
            terminals = torch.tensor([False], dtype=torch.float),
            discounts = torch.tensor([0.9], dtype=torch.float),
            gae_lambda = 1,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == approx(1+0.9*6 - 5)

def test_1_steps_terminal_gae_1():
    output = gae(
            state_values = torch.tensor([5], dtype=torch.float),
            next_state_values = torch.tensor([6], dtype=torch.float),
            rewards = torch.tensor([1], dtype=torch.float),
            terminals = torch.tensor([True], dtype=torch.float),
            discounts = torch.tensor([0.9], dtype=torch.float),
            gae_lambda = 1,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == approx(1 - 5)

def test_2_steps_gae_1():
    output = gae(
            state_values = torch.tensor([5,6], dtype=torch.float),
            next_state_values = torch.tensor([6,7], dtype=torch.float),
            rewards = torch.tensor([1,2], dtype=torch.float),
            terminals = torch.tensor([False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9], dtype=torch.float),
            gae_lambda = 1,
    )
    assert torch.tensor(output.shape).tolist() == [2]
    d1 = 2+0.9*7 - 6
    assert output[1].item() == approx(d1)
    d0 = 1+0.9*(2+0.9*7) - 5
    assert output[0].item() == approx(d0)

def test_3_steps_gae_1():
    output = gae(
            state_values = torch.tensor([5,6,7], dtype=torch.float),
            next_state_values = torch.tensor([6,7,8], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            gae_lambda = 1,
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
            state_values = torch.tensor([5,6,7], dtype=torch.float),
            next_state_values = torch.tensor([6,7,8], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([True,False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            gae_lambda = 1,
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
            state_values = torch.tensor([5,6,7], dtype=torch.float),
            next_state_values = torch.tensor([6,7,8], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,False,True], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            gae_lambda = 1,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    d2 = 3 - 7
    assert output[2].item() == approx(d2)
    d1 = 2+0.9*(3) - 6
    assert output[1].item() == approx(d1)
    d0 = 1+0.9*(2+0.9*(3)) - 5
    assert output[0].item() == approx(d0,1e-5)

def test_termination_in_middle_gae_1():
    output = gae(
            state_values = torch.tensor([5,6,7], dtype=torch.float),
            next_state_values = torch.tensor([6,7,8], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,True,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            gae_lambda = 1,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    d2 = 3+0.9*8 - 7
    assert output[2].item() == approx(d2)
    d1 = 2 - 6
    assert output[1].item() == approx(d1)
    d0 = 1+0.9*(2) - 5
    assert output[0].item() == approx(d0)
