from pytest import approx
import torch

from frankenstein.loss.policy_gradient import advantage_policy_gradient_loss as loss
from frankenstein.advantage.gae import generalized_advantage_estimate as gae

def test_0_steps():
    output = loss(
            log_action_probs = torch.tensor([], dtype=torch.float),
            terminals = torch.tensor([], dtype=torch.float),
            advantages = torch.tensor([], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [0]

def test_1_steps():
    output = loss(
            log_action_probs = torch.tensor([-1], dtype=torch.float),
            terminals = torch.tensor([False], dtype=torch.float),
            advantages = torch.tensor([0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [1]
    assert output.item() == approx(0.9)

def test_3_steps():
    output = loss(
            log_action_probs = torch.tensor([-1,-2,-3], dtype=torch.float),
            terminals = torch.tensor([False,False,False], dtype=torch.float),
            advantages = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[0].item() == approx(0.9)
    assert output[1].item() == approx(1.8)
    assert output[2].item() == approx(2.7)

# Make sure it learns the correct policy
def test_training():
    # Test on a 1 state environment. Reward is different for each action.
    # Action 1 gives a reward of 1, action 0 gives a reward of 0
    policy_weights = torch.tensor([0,0], dtype=torch.float, requires_grad=True)
    optimizer = torch.optim.SGD([policy_weights], lr=0.1)
    for _ in range(5):
        probs = policy_weights.log_softmax(0)
        state_value = torch.tensor([probs[1].exp()/(1-0.9)])
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        advantages = gae(
                state_values = state_value,
                next_state_values = state_value,
                rewards = action.unsqueeze(0),
                terminals = torch.tensor([False]),
                discount = 0.9,
                gae_lambda=0.95,
        )
        l = loss(
                log_action_probs = probs[action],
                terminals = torch.tensor([False]),
                advantages = advantages,
        )
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    # The probability for the good action should be higher than that of the bad action
    assert policy_weights[1] > policy_weights[0]
