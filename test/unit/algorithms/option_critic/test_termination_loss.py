import torch

from frankenstein.algorithms.option_critic import compute_termination_loss as loss, compute_termination_loss_batch as loss_batch


def test_zero_prob():
    termination_prob = torch.tensor([0.])
    option_values_current = torch.tensor([1])
    option_values_max = torch.tensor([2])
    output = loss(
            termination_prob=termination_prob,
            option_values_current=option_values_current,
            option_values_max=option_values_max,
            termination_reg=0,
            deliberation_cost=0,
    )
    assert output.item() == 0


def test_termination_prob_unchanged():
    """ If the value of the current option and the best option are the same, then the termination prob should not change. """
    termination_prob = torch.tensor([0.5], requires_grad=True)
    option_values_current = torch.tensor([1])
    option_values_max = torch.tensor([1])
    l = loss(
            termination_prob=termination_prob,
            option_values_current=option_values_current,
            option_values_max=option_values_max,
            termination_reg=0,
            deliberation_cost=0
    )
    l.backward()
    assert l.item() == 0
    assert termination_prob.grad is not None
    assert termination_prob.grad.item() == 0


def test_termination_prob_increases():
    """ If the value of the current option is less than the best option, then the termination prob should increase. """
    termination_prob = torch.tensor([0.5], requires_grad=True)
    option_values_current = torch.tensor([1])
    option_values_max = torch.tensor([2])
    l = loss(
            termination_prob=termination_prob,
            option_values_current=option_values_current,
            option_values_max=option_values_max,
            termination_reg=0,
            deliberation_cost=0,
    )
    l.backward()
    assert termination_prob.grad is not None
    assert termination_prob.grad.item() < 0


def test_termination_prob_decreases():
    """ If the value of the current option is greater than the best option, then the termination prob should decrease. """
    termination_prob = torch.tensor([0.5], requires_grad=True)
    option_values_current = torch.tensor([2])
    option_values_max = torch.tensor([1])
    l = loss(
            termination_prob=termination_prob,
            option_values_current=option_values_current,
            option_values_max=option_values_max,
            termination_reg=0,
            deliberation_cost=0,
    )
    l.backward()
    assert termination_prob.grad is not None
    assert termination_prob.grad.item() > 0


def test_termination_prob_decreases_same_val():
    """ If the value of the current option and the best option are the same, and a non-zero deliberation cost is given, then the termination prob should decrease in order to increase the duration of the current option. """
    termination_prob = torch.tensor([0.5], requires_grad=True)
    option_values_current = torch.tensor([1])
    option_values_max = torch.tensor([1])
    l = loss(
            termination_prob=termination_prob,
            option_values_current=option_values_current,
            option_values_max=option_values_max,
            termination_reg=0,
            deliberation_cost=0.01
    )
    l.backward()
    assert termination_prob.grad is not None
    assert termination_prob.grad.item() > 0


def test_sequence():
    """ The loss computed for a sequence should be the same as the loss computed for each individual step. """
    seq_len = 5
    termination_prob = torch.tensor([0.5] * seq_len, requires_grad=True)
    option_values_current = torch.rand(seq_len)
    option_values_max = torch.rand(seq_len)
    l = loss(
            termination_prob=termination_prob,
            option_values_current=option_values_current,
            option_values_max=option_values_max,
            termination_reg=0.01,
            deliberation_cost=0.02,
    )
    l.backward()
    assert termination_prob.grad is not None

    total_loss = []
    termination_prob2 = torch.tensor([0.5] * seq_len, requires_grad=True)
    for i in range(seq_len):
        l2 = loss(
                termination_prob=termination_prob2[i:i+1],
                option_values_current=option_values_current[i:i+1],
                option_values_max=option_values_max[i:i+1],
                termination_reg=0.01,
                deliberation_cost=0.02,
        )
        total_loss.append(l2)
    torch.stack(total_loss).mean().backward()
    assert termination_prob2.grad is not None
    assert (termination_prob2.grad == termination_prob.grad).all()


def test_batch():
    """ The loss computed for a batch should be the same as the loss computed for each individual sequence. """
    seq_len = 5
    batch_size = 10
    termination_prob = torch.tensor([[0.5] * batch_size] * seq_len, requires_grad=True)
    option_values_current = torch.rand(seq_len,batch_size)
    option_values_max = torch.rand(seq_len,batch_size)
    l = loss_batch(
            termination_prob=termination_prob,
            option_values_current=option_values_current,
            option_values_max=option_values_max,
            termination_reg=0.01,
            deliberation_cost=0.02,
    )
    l.backward()
    assert termination_prob.grad is not None

    total_loss = []
    termination_prob2 = torch.tensor([[0.5] * batch_size] * seq_len, requires_grad=True)
    for i in range(batch_size):
        l2 = loss(
                termination_prob=termination_prob2[:,i:i+1],
                option_values_current=option_values_current[:,i:i+1],
                option_values_max=option_values_max[:,i:i+1],
                termination_reg=0.01,
                deliberation_cost=0.02,
        )
        total_loss.append(l2)
    torch.stack(total_loss).mean().backward()
    assert termination_prob2.grad is not None
    assert (termination_prob2.grad == termination_prob.grad).all()
