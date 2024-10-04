from pytest import approx

import torch

from frankenstein.loss.value import clipped_value_loss


def test_empty():
    output = clipped_value_loss(
            state_values = torch.tensor([]),
            state_values_old = torch.tensor([]),
            returns = torch.tensor([]),
            clip_vf_loss = 0.1,
    )
    assert torch.tensor(output.shape).tolist() == [0]


def test_no_clip():
    output = clipped_value_loss(
            state_values = torch.tensor([1.]),
            state_values_old = torch.tensor([1.]),
            returns = torch.tensor([2.]),
            clip_vf_loss = 0.1,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == 0.5


def test_no_clip_below_threshold():
    output = clipped_value_loss(
            state_values = torch.tensor([1.]),
            state_values_old = torch.tensor([1.1]),
            returns = torch.tensor([2.]),
            clip_vf_loss = 0.1,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == 0.5


def test_no_clip_because_clipped_is_larger():
    output = clipped_value_loss(
            state_values = torch.tensor([1.]),
            state_values_old = torch.tensor([1.2]),
            returns = torch.tensor([2.]),
            clip_vf_loss = 0.1,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == 0.5


def test_clip():
    output = clipped_value_loss(
            state_values = torch.tensor([1.]),
            state_values_old = torch.tensor([0.8]),
            returns = torch.tensor([2.]),
            clip_vf_loss = 0.1,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == approx(0.5 * (0.9 - 2) ** 2)


def test_batch():
    batch_size = 10
    state_values = torch.rand(batch_size)
    state_values_old = torch.rand(batch_size)
    returns = torch.rand(batch_size)
    output_batch = clipped_value_loss(
            state_values = state_values,
            state_values_old = state_values_old,
            returns = returns,
            clip_vf_loss = 0.1,
    )
    assert torch.tensor(output_batch.shape).tolist() == [batch_size]

    for i in range(batch_size):
        output = clipped_value_loss(
                state_values = state_values[i],
                state_values_old = state_values_old[i],
                returns = returns[i],
                clip_vf_loss = 0.1,
        )
        assert output_batch[i].item() == approx(output.item())
