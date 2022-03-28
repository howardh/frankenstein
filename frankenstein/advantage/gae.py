import torch
from torchtyping import TensorType


def generalized_advantage_estimate(
        state_values: TensorType['num_steps', float],
        next_state_values: TensorType['num_steps', float],
        rewards: TensorType['num_steps', float],
        terminals: TensorType['num_steps', bool],
        discount: float,
        gae_lambda: float,
) -> TensorType['num_steps', float]:
    """
    Return the n-step value of each state.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        state_values: A tensor containing $[V(s_0),V_(s_1),...,V(s_{n-1})]$ where $V(s)$ is the estimated value of state $s$.
        next_state_values: A tensor containing $[V(s_1),V_(s_2),...,V(s_n)]$ where $V(s)$ is the estimated value of state $s$.
        rewards: A tensor containing $[r_1,r_2,...,r_n]$.
        terminals: A boolean tensor containing $[T(s_1),T(s_2),...,T(s_n)]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discount: Discount factor.
        gae_lambda: ...
    """
    device = state_values.device
    num_steps = state_values.shape[0]
    if num_steps == 0:
        return torch.tensor([], dtype=torch.float, device=device)
    adv = torch.zeros([num_steps+1], device=device)
    delta = rewards+discount*next_state_values*terminals.logical_not()-state_values
    for i in reversed(range(num_steps)):
        adv[i] = delta[i]+terminals[i].logical_not()*gae_lambda*discount*adv[i+1]
    return adv[:-1]


def generalized_advantage_estimate_batch(
        state_values: TensorType['num_steps', 'batch_size', float],
        next_state_values: TensorType['num_steps', 'batch_size', float],
        rewards: TensorType['num_steps', 'batch_size', float],
        terminals: TensorType['num_steps', 'batch_size', bool],
        discount: float,
        gae_lambda: float,
) -> TensorType['batch_size', 'num_steps', float]:
    """
    Return the n-step value of each state.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        state_values: A tensor containing $[V(s_0),V_(s_1),...,V(s_{n-1})]$ where $V(s)$ is the estimated value of state $s$.
        next_state_values: A tensor containing $[V(s_1),V_(s_2),...,V(s_n)]$ where $V(s)$ is the estimated value of state $s$.
        rewards: A tensor containing $[r_1,r_2,...,r_n]$.
        terminals: A boolean tensor containing $[T(s_1),T(s_2),...,T(s_n)]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discount: Discount factor.
        gae_lambda: ...
    """
    device = state_values.device
    num_steps = state_values.shape[0]
    if num_steps == 0:
        return torch.empty_like(next_state_values, dtype=torch.float, device=device)
    batch_size = state_values.shape[1]
    adv = torch.zeros([num_steps+1, batch_size], device=device)
    delta = rewards+discount*next_state_values*terminals.logical_not()-state_values
    for i in reversed(range(num_steps)):
        adv[i,:] = delta[i,:]+terminals[i,:].logical_not()*gae_lambda*discount*adv[i+1,:]
    return adv[:-1,:]
