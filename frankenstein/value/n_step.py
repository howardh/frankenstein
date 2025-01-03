import torch
from jaxtyping import Float, Bool


def n_step_return_iterative(
    next_state_values: Float[torch.Tensor, 'num_steps'],
    rewards: Float[torch.Tensor, 'num_steps'],
    terminals: Bool[torch.Tensor, 'num_steps'],
    discount: float,
    n: int,
) -> Float[torch.Tensor, 'num_steps']:
    """
    Return the $n$-step return of each state.
    The output is computed in an interative manner, so this is less computationally efficient, but the code may be easier to understand.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_m,s_m,a_m)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        next_state_values: A tensor containing $[V(s_1),V_(s_2),...,V(s_m)]$ where $V(s)$ is the estimated value of state $s$.
        rewards: A tensor containing $[r_1,r_2,...,r_m]$.
        terminals: A boolean tensor containing $[T(s_1),T(s_2),...,T(s_m)]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discounts: A tensor $[\\gamma_1,\\gamma_2,...,\\gamma_m]$ where the 1-step value estimate of $s_0$ is $r_1+\\gamma_1 V(s_1)$.

    Returns:
        A tensor $[G_{0:n}, G_{1:1+n}, \\cdots, G_{m-n:m}]$ (Using the notation from Sutton&Barto Intro to RL) where
        $$G_{i:i+n} = \\left[\\sum_{j=1}^n \\gamma^{j-1} r_{i+j}\\right] + V(s_{i+n})$$
    """
    assert n >= 1

    device = next_state_values.device
    num_steps = next_state_values.shape[0]
    output = torch.zeros([num_steps-n+1], device=device)
    for i in range(num_steps-n+1):
        ret = 0.
        for j in range(i, i+n):
            ret += (discount**(j-i))*rewards[j]
            if terminals[j]:
                break
        else:  # If we finish the loop and didn't break out
            ret += (discount**n)*next_state_values[i+n-1]
        output[i] = ret
    return output
