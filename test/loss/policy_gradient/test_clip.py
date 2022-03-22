import torch

from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss as loss
from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss_batch as loss_batch
from frankenstein.loss.policy_gradient import advantage_policy_gradient_loss as loss_adv


def test_0_steps():
    output = loss(
            log_action_probs = torch.tensor([], dtype=torch.float),
            old_log_action_probs = torch.tensor([], dtype=torch.float),
            state_values = torch.tensor([], dtype=torch.float),
            next_state_values = torch.tensor([], dtype=torch.float),
            rewards = torch.tensor([], dtype=torch.float),
            terminals = torch.tensor([], dtype=torch.float),
            prev_terminals = torch.tensor([], dtype=torch.float),
            discounts = torch.tensor([], dtype=torch.float),
            epsilon=0.1,
    )
    assert torch.tensor(output.shape).tolist() == [0]


# Check that the gradient is the same as the standard advantage policy gradient when there is no clipping


def test_1_steps_matches_advantage_pg():
    action_probs = torch.tensor([-1], dtype=torch.float, requires_grad=True)
    output = loss(
            log_action_probs = action_probs,
            old_log_action_probs = action_probs.detach(),
            state_values = torch.tensor([4], dtype=torch.float),
            next_state_values = torch.tensor([5], dtype=torch.float),
            rewards = torch.tensor([1], dtype=torch.float),
            terminals = torch.tensor([False], dtype=torch.float),
            prev_terminals = torch.tensor([False], dtype=torch.float),
            discounts = torch.tensor([0.9], dtype=torch.float),
            epsilon=1000, # Large number, so the ratio won't be clipped
    )
    action_probs_adv = torch.tensor([-1], dtype=torch.float, requires_grad=True)
    output_adv = loss_adv(
            log_action_probs = action_probs_adv,
            state_values = torch.tensor([4], dtype=torch.float),
            next_state_values = torch.tensor([5], dtype=torch.float),
            rewards = torch.tensor([1], dtype=torch.float),
            terminals = torch.tensor([False], dtype=torch.float),
            prev_terminals = torch.tensor([False], dtype=torch.float),
            discounts = torch.tensor([0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [1]
    output.mean().backward()
    output_adv.mean().backward()
    assert action_probs.grad == action_probs_adv.grad


def test_3_steps_matches_advantage_pg():
    action_probs = torch.tensor([-1,-2,-3], dtype=torch.float, requires_grad=True)
    output = loss(
            log_action_probs = action_probs,
            old_log_action_probs = action_probs.detach(),
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,False,False], dtype=torch.float),
            prev_terminals = torch.tensor([False,False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            epsilon=1000, # Large number, so the ratio won't be clipped
    )
    action_probs_adv = torch.tensor([-1,-2,-3], dtype=torch.float, requires_grad=True)
    output_adv = loss_adv(
            log_action_probs = action_probs_adv,
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,False,False], dtype=torch.float),
            prev_terminals = torch.tensor([False,False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    output.mean().backward()
    output_adv.mean().backward()
    assert action_probs.grad is not None
    assert action_probs_adv.grad is not None
    assert torch.isclose(action_probs.grad, action_probs_adv.grad).all()


def test_termination_in_middle_matches_advantage_pg():
    action_probs = torch.tensor([-1,-2,-3], dtype=torch.float, requires_grad=True)
    output = loss(
            log_action_probs = action_probs,
            old_log_action_probs = action_probs.detach(),
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,True,False], dtype=torch.float),
            prev_terminals = torch.tensor([False,False,True], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            epsilon=1000, # Large number, so the ratio won't be clipped
    )
    action_probs_adv = torch.tensor([-1,-2,-3], dtype=torch.float, requires_grad=True)
    output_adv = loss_adv(
            log_action_probs = action_probs_adv,
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,True,False], dtype=torch.float),
            prev_terminals = torch.tensor([False,False,True], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    output.mean().backward()
    output_adv.mean().backward()
    assert action_probs.grad is not None
    assert action_probs_adv.grad is not None
    assert torch.isclose(action_probs.grad, action_probs_adv.grad).all()


# Make sure the batch version behaves the same as the non-batched version
def test_batch_matches_non_batched():
    num_steps = 5
    batch_size = 10
    log_action_probs = torch.rand([num_steps,batch_size]).log_softmax(1)
    old_log_action_probs = torch.rand([num_steps,batch_size]).log_softmax(1)
    state_values = torch.rand([num_steps+1,batch_size])
    rewards = torch.rand([num_steps,batch_size])
    terminals = (torch.rand([num_steps+1,batch_size])*2).floor().bool()
    discounts = torch.rand([num_steps,batch_size])
    epsilon = 0.1
    output_batch = loss_batch(
            log_action_probs=log_action_probs,
            old_log_action_probs=old_log_action_probs,
            state_values = state_values[:-1,:],
            next_state_values = state_values[1:,:],
            rewards = rewards,
            terminals = terminals[1:,:],
            prev_terminals = terminals[:-1,:],
            discounts = discounts,
            epsilon=epsilon,
    )
    for i in range(batch_size):
        output = loss(
                log_action_probs=log_action_probs[:,i],
                old_log_action_probs=old_log_action_probs[:,i],
                state_values = state_values[:-1,i],
                next_state_values = state_values[1:,i],
                rewards = rewards[:,i],
                terminals = terminals[1:,i],
                prev_terminals = terminals[:-1,i],
                discounts = discounts[:,i],
                epsilon=epsilon,
        )
        assert torch.isclose(output, output_batch[:,i]).all()
