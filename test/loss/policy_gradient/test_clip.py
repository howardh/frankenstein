import torch

from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss as loss
from frankenstein.loss.policy_gradient import advantage_policy_gradient_loss as loss_adv

def test_0_steps():
    output = loss(
            action_probs = torch.tensor([], dtype=torch.float),
            old_action_probs = torch.tensor([], dtype=torch.float),
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
    action_probs = torch.tensor([0.5], dtype=torch.float, requires_grad=True)
    output = loss(
            action_probs = action_probs,
            old_action_probs = action_probs.detach(),
            state_values = torch.tensor([4], dtype=torch.float),
            next_state_values = torch.tensor([5], dtype=torch.float),
            rewards = torch.tensor([1], dtype=torch.float),
            terminals = torch.tensor([False], dtype=torch.float),
            prev_terminals = torch.tensor([False], dtype=torch.float),
            discounts = torch.tensor([0.9], dtype=torch.float),
            epsilon=1000, # Large number, so the ratio won't be clipped
    )
    action_probs_adv = torch.tensor([0.5], dtype=torch.float, requires_grad=True)
    output_adv = loss_adv(
            log_action_probs = action_probs_adv.log(),
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
    action_probs = torch.tensor([0.5,0.4,0.3], dtype=torch.float, requires_grad=True)
    output = loss(
            action_probs = action_probs,
            old_action_probs = action_probs.detach(),
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,False,False], dtype=torch.float),
            prev_terminals = torch.tensor([False,False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            epsilon=1000, # Large number, so the ratio won't be clipped
    )
    action_probs_adv = torch.tensor([0.5,0.4,0.3], dtype=torch.float, requires_grad=True)
    output_adv = loss_adv(
            log_action_probs = action_probs_adv.log(),
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
    action_probs = torch.tensor([0.5,0.4,0.3], dtype=torch.float, requires_grad=True)
    output = loss(
            action_probs = action_probs,
            old_action_probs = action_probs.detach(),
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,True,False], dtype=torch.float),
            prev_terminals = torch.tensor([False,False,True], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
            epsilon=1000, # Large number, so the ratio won't be clipped
    )
    action_probs_adv = torch.tensor([0.5,0.4,0.3], dtype=torch.float, requires_grad=True)
    output_adv = loss_adv(
            log_action_probs = action_probs_adv.log(),
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
