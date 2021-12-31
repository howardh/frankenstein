from pytest import approx

import torch

from frankenstein.value.n_step import n_step_return_iterative

def test_1_step_return_0_steps():
    output = n_step_return_iterative(
            next_state_values=torch.tensor([]),
            rewards=torch.tensor([]),
            terminals=torch.tensor([]),
            discount=0.9,
            n=1,
    )
    assert len(output) == 0

def test_1_step_return_1_step():
    output = n_step_return_iterative(
            next_state_values=torch.tensor([10]),
            rewards=torch.tensor([1]),
            terminals=torch.tensor([False]),
            discount=0.9,
            n=1,
    )
    assert len(output) == 1
    assert output[0].item() == approx(1+0.9*10)

def test_1_step_return_3_steps():
    output = n_step_return_iterative(
            next_state_values=torch.tensor([10,11,12]),
            rewards=torch.tensor([1,2,3]),
            terminals=torch.tensor([False,False,False]),
            discount=0.9,
            n=1,
    )
    assert len(output) == 3
    assert output[0].item() == approx(1+0.9*10)
    assert output[1].item() == approx(2+0.9*11)
    assert output[2].item() == approx(3+0.9*12)

def test_1_step_return_3_steps_with_termination():
    output = n_step_return_iterative(
            next_state_values=torch.tensor([10,11,12]),
            rewards=torch.tensor([1,2,3]),
            terminals=torch.tensor([False,True,False]),
            discount=0.9,
            n=1,
    )
    assert len(output) == 3
    assert output[0].item() == approx(1+0.9*10)
    assert output[1].item() == approx(2)
    assert output[2].item() == approx(3+0.9*12)

def test_2_step_return_3_steps():
    output = n_step_return_iterative(
            next_state_values=torch.tensor([10,11,12]),
            rewards=torch.tensor([1,2,3]),
            terminals=torch.tensor([False,False,False]),
            discount=0.9,
            n=2,
    )
    assert len(output) == 2
    assert output[0].item() == approx(1+0.9*(2+0.9*11))
    assert output[1].item() == approx(2+0.9*(3+0.9*12))

def test_3_step_return_3_steps():
    output = n_step_return_iterative(
            next_state_values=torch.tensor([10,11,12]),
            rewards=torch.tensor([1,2,3]),
            terminals=torch.tensor([False,False,False]),
            discount=0.9,
            n=3,
    )
    assert len(output) == 1
    assert output[0].item() == approx(1+0.9*(2+0.9*(3+0.9*12)))
