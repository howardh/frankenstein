import torch

from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss as loss
from frankenstein.loss.policy_gradient import advantage_policy_gradient_loss as loss_adv


def test_0_steps():
    output = loss(
            old_log_action_probs = torch.tensor([], dtype=torch.float),
            log_action_probs = torch.tensor([], dtype=torch.float),
            terminals = torch.tensor([], dtype=torch.float),
            advantages = torch.tensor([], dtype=torch.float),
            epsilon=0.1,
    )
    assert torch.tensor(output.shape).tolist() == [0]


# Check that the gradient is the same as the standard advantage policy gradient when there is no clipping


def test_1_steps_matches_advantage_pg():
    action_probs = torch.tensor([-1], dtype=torch.float, requires_grad=True)
    advantages = torch.tensor([1], dtype=torch.float, requires_grad=True)
    output = loss(
            log_action_probs = action_probs,
            old_log_action_probs = action_probs.detach(),
            terminals = torch.tensor([False], dtype=torch.float),
            advantages=advantages,
            epsilon=1000, # Large number, so the ratio won't be clipped
    )
    action_probs_adv = torch.tensor([-1], dtype=torch.float, requires_grad=True)
    output_adv = loss_adv(
            log_action_probs = action_probs_adv,
            terminals = torch.tensor([False], dtype=torch.float),
            advantages = advantages,
    )
    assert torch.tensor(output.shape).tolist() == [1]
    output.mean().backward()
    output_adv.mean().backward()
    assert action_probs.grad == action_probs_adv.grad


def test_3_steps_matches_advantage_pg():
    action_probs = torch.tensor([-1, -2, -3], dtype=torch.float, requires_grad=True)
    advantages = torch.tensor([1, 2, 3], dtype=torch.float, requires_grad=True)
    output = loss(
            log_action_probs = action_probs,
            old_log_action_probs = action_probs.detach(),
            terminals = torch.tensor([False, False, False], dtype=torch.float),
            advantages=advantages,
            epsilon=1000, # Large number, so the ratio won't be clipped
    )
    action_probs_adv = torch.tensor([-1, -2, -3], dtype=torch.float, requires_grad=True)
    output_adv = loss_adv(
            log_action_probs = action_probs_adv,
            terminals = torch.tensor([False, False, False], dtype=torch.float),
            advantages = advantages,
    )
    assert torch.tensor(output.shape).tolist() == [3]
    output.mean().backward()
    output_adv.mean().backward()
    assert action_probs.grad is not None
    assert action_probs_adv.grad is not None
    assert torch.isclose(action_probs.grad, action_probs_adv.grad).all()
