from pytest import approx

import torch

from frankenstein.value.lam import lambda_return_iterative as lambda_return


def test_lambda_0_with_1_steps():
    output = lambda_return(
        next_state_values=torch.tensor([10]),
        rewards=torch.tensor([1]),
        terminals=torch.tensor([False]),
        discount=0.9,
        lam=0,
    )
    assert len(output) == 1
    assert output[0].item() == approx(1+0.9*10)


def test_lambda_0_with_3_steps():
    output = lambda_return(
        next_state_values=torch.tensor([10, 11, 12]),
        rewards=torch.tensor([1, 2, 3]),
        terminals=torch.tensor([False, False, False]),
        discount=0.9,
        lam=0,
    )
    assert len(output) == 3
    assert output[0].item() == approx(1+0.9*10)
    assert output[1].item() == approx(2+0.9*11)
    assert output[2].item() == approx(3+0.9*12)


def test_lambda_1_with_3_steps():
    output = lambda_return(
        next_state_values=torch.tensor([10, 11, 12]),
        rewards=torch.tensor([1, 2, 3]),
        terminals=torch.tensor([False, False, False]),
        discount=0.9,
        lam=1,
    )
    assert len(output) == 3
    assert output[0].item() == approx(1+0.9*(2+0.9*(3+0.9*12)))
    assert output[1].item() == approx(2+0.9*(3+0.9*12))
    assert output[2].item() == approx(3+0.9*12)


def test_lambda_0_5_with_5_steps():
    output = lambda_return(
        next_state_values=torch.tensor([10, 11, 12, 13, 14]),
        rewards=torch.tensor([1, 2, 3, 4, 5]),
        terminals=torch.tensor([False, False, False, False, False]),
        discount=0.9,
        lam=0.5,
    )
    assert len(output) == 5
    ret5 = 1+0.9*(2+0.9*(3+0.9*(4+0.9*(5+0.9*14))))     # 5-step return
    ret4 = 1+0.9*(2+0.9*(3+0.9*(4+0.9*13)))             # ...
    ret3 = 1+0.9*(2+0.9*(3+0.9*12))                     # ...
    ret2 = 1+0.9*(2+0.9*11)                             # 2-step return
    ret1 = 1+0.9*10                                     # 1-step return
    assert output[0].item() == approx(
        (1-0.5)*ret1 +
        (1-0.5)*0.5*ret2 +
        (1-0.5)*0.5*0.5*ret3 +
        (1-0.5)*0.5*0.5*0.5*ret4 +
        0.5*0.5*0.5*0.5*ret5
    )

    assert output[4].item() == approx(5+0.9*14)
