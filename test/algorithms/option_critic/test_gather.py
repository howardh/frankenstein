import torch

from frankenstein.algorithms.option_critic import gather_option_action as gather
from frankenstein.algorithms.option_critic import gather_option_action_batch as gather_batch


def test_seq_len_1():
    num_options = 2
    num_actions = 3
    data = torch.tensor([[1, 2, 3], [4, 5, 6]]).reshape(1, num_options, num_actions)

    option = torch.tensor([[0]])
    action = torch.tensor([[0]])
    output = gather(data, option, action)
    assert output.shape == (1,)
    assert output.item() == 1

    option = torch.tensor([[1]])
    action = torch.tensor([[1]])
    output = gather(data, option, action)
    assert output.shape == (1,)
    assert output.item() == 5

    option = torch.tensor([[0]])
    action = torch.tensor([[2]])
    output = gather(data, option, action)
    assert output.shape == (1,)
    assert output.item() == 3


def test_batch():
    """ The batched version should behave the same as the non-batched version """
    seq_len = 5
    batch_size = 10
    num_options = 2
    num_actions = 3
    data = torch.rand([seq_len, batch_size, num_options, num_actions])
    option = torch.randint(0, num_options, [seq_len, batch_size])
    action = torch.randint(0, num_actions, [seq_len, batch_size])

    output_batch = gather_batch(data, option, action)
    output_no_batch = torch.stack([
        gather(data[:, i, :, :], option[:, i], action[:, i])
        for i in range(batch_size)
    ], dim=1)

    assert output_batch.shape == output_no_batch.shape
    assert (output_batch == output_no_batch).all()
