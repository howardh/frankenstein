from pytest import approx
import torch

from frankenstein.loss.policy_gradient import advantage_policy_gradient_loss as loss

def test_0_steps():
    output = loss(
            log_action_probs = torch.tensor([], dtype=torch.float),
            state_values = torch.tensor([], dtype=torch.float),
            next_state_values = torch.tensor([], dtype=torch.float),
            rewards = torch.tensor([], dtype=torch.float),
            terminals = torch.tensor([], dtype=torch.float),
            prev_terminals = torch.tensor([], dtype=torch.float),
            discounts = torch.tensor([], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [0]

def test_1_steps():
    output = loss(
            log_action_probs = torch.tensor([-1], dtype=torch.float),
            state_values = torch.tensor([4], dtype=torch.float),
            next_state_values = torch.tensor([5], dtype=torch.float),
            rewards = torch.tensor([1], dtype=torch.float),
            terminals = torch.tensor([False], dtype=torch.float),
            prev_terminals = torch.tensor([False], dtype=torch.float),
            discounts = torch.tensor([0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [1]
    v = 1+0.9*5
    assert output.item() == approx(1*(v-4))

def test_3_steps():
    output = loss(
            log_action_probs = torch.tensor([-1,-2,-3], dtype=torch.float),
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,False,False], dtype=torch.float),
            prev_terminals = torch.tensor([False,False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    v2 = 3+0.9*7
    assert output[2].item() == approx(3*(v2-6))
    v1 = 2+0.9*v2
    assert output[1].item() == approx(2*(v1-5))
    v0 = 1+0.9*v1
    assert output[0].item() == approx(1*(v0-4))

# With termination
# Same discount for all steps
def test_termination_at_start():
    output = loss(
            log_action_probs = torch.tensor([-1,-2,-3], dtype=torch.float),
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([True,False,False], dtype=torch.float),
            prev_terminals = torch.tensor([False,True,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    v2 = 3+0.9*7
    assert output[2].item() == approx(3*(v2-6))
    assert output[1].item() == approx(0)
    v0 = 1
    assert output[0].item() == approx(1*(v0-4))

def test_termination_at_end():
    output = loss(
            log_action_probs = torch.tensor([-1,-2,-3], dtype=torch.float),
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,False,True], dtype=torch.float),
            prev_terminals = torch.tensor([False,False,False], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    v2 = 3
    assert output[2].item() == approx(3*(v2-6))
    v1 = 2+0.9*v2
    assert output[1].item() == approx(2*(v1-5))
    v0 = 1+0.9*v1
    assert output[0].item() == approx(1*(v0-4))

def test_termination_in_middle():
    output = loss(
            log_action_probs = torch.tensor([-1,-2,-3], dtype=torch.float),
            state_values = torch.tensor([4,5,6], dtype=torch.float),
            next_state_values = torch.tensor([5,6,7], dtype=torch.float),
            rewards = torch.tensor([1,2,3], dtype=torch.float),
            terminals = torch.tensor([False,True,False], dtype=torch.float),
            prev_terminals = torch.tensor([False,False,True], dtype=torch.float),
            discounts = torch.tensor([0.9,0.9,0.9], dtype=torch.float),
    )
    assert torch.tensor(output.shape).tolist() == [3]
    assert output[2].item() == approx(0)
    v1 = 2
    assert output[1].item() == approx(2*(v1-5))
    v0 = 1+0.9*v1
    assert output[0].item() == approx(1*(v0-4))

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
        l = loss(
                log_action_probs = probs[action],
                state_values = state_value,
                next_state_values = state_value,
                rewards = action.unsqueeze(0),
                terminals = torch.tensor([False]),
                prev_terminals = torch.tensor([False]),
                discounts = torch.tensor([0.9]),
        )
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    # The probability for the good action should be higher than that of the bad action
    assert policy_weights[1] > policy_weights[0]
