import torch
from jaxtyping import Float, Bool


def generalized_advantage_estimate(
        state_values: Float[torch.Tensor, 'num_steps batch_size'],
        next_state_values: Float[torch.Tensor, 'num_steps batch_size'],
        rewards: Float[torch.Tensor, 'num_steps batch_size'],
        terminals: Bool[torch.Tensor, 'num_steps batch_size'],
        discount: float,
        gae_lambda: float,
) -> Float[torch.Tensor, 'batch_size num_steps']:
    """
    Return the n-step value of each state.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    The inputs can be of any shape as long as the first dimension is time.

    Args:
        state_values: A tensor containing $[V(s_0),V_(s_1),...,V(s_{n-1})]$ where $V(s)$ is the estimated value of state $s$.
        next_state_values: A tensor containing $[V(s_1),V_(s_2),...,V(s_n)]$ where $V(s)$ is the estimated value of state $s$.
        rewards: A tensor containing $[r_1,r_2,...,r_n]$.
        terminals: A boolean tensor containing $[T(s_1),T(s_2),...,T(s_n)]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discount: Discount factor.
        gae_lambda: ...

    Returns:
        A tensor containing $[A^\\pi(s_0,a_0), A^\\pi(s_1,a_1),\\cdots, A^\\pi(s_{n-1},a_{n-1})]$ where $A^\\pi(s,a)=Q^\\pi(s,a)-V^\\pi(s)$ is the advantage of taking action $a$ at state $s$ over following policy $\\pi$. If $s_i$ is a terminal state, then 
    """
    device = state_values.device
    num_steps = state_values.shape[0]
    if num_steps == 0:
        return torch.empty_like(next_state_values, dtype=torch.float, device=device)
    adv = torch.zeros([num_steps+1, *state_values.shape[1:]], device=device)
    delta = rewards+discount*next_state_values*terminals.logical_not()-state_values
    for i in reversed(range(num_steps)):
        adv[i] = delta[i]+terminals[i].logical_not()*gae_lambda*discount*adv[i+1]
    return adv[:-1]
