import torch
from torchtyping import TensorType

from frankenstein.value.n_step import n_step_return_iterative

def lambda_return_iterative(
        next_state_values : TensorType['num_steps',float],
        rewards : TensorType['num_steps',float],
        terminals : TensorType['num_steps',bool],
        discount: float,
        lam : float,
    ) -> TensorType['num_steps',float]:
    """
    Return the truncated $\\lambda$ return of each state.
    The output is computed in an interative manner, so this is less computationally efficient, but the code may be easier to understand.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        next_state_values: A tensor containing $[V(s_1),V_(s_2),...,V(s_n)]$ where $V(s)$ is the estimated value of state $s$.
        rewards: A tensor containing $[r_1,r_2,...,r_n]$.
        terminals: A boolean tensor containing $[T(s_1),T(s_2),...,T(s_n)]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discounts: A tensor $[\\gamma_1,\\gamma_2,...,\\gamma_n]$ where the 1-step value estimate of $s_0$ is $r_1+\\gamma_1 V(s_1)$.
        lam (float): $\\lambda$

    Returns:
        A tensor $[G^\\lambda_{0:n}, G^\\lambda_{1:n}, \\cdots, G^\\lambda_{n-1:n}]$ (Using the notation from Sutton&Barto Intro to RL) where
        $$G^\\lambda_{i:n} = \\left[(1-\\lambda)\\sum_{j=1}^{n-j} \\lambda^{j-1}G_{i:i+j} \\right] + ...$$
    """
    device = next_state_values.device
    num_steps = next_state_values.shape[0]
    output = torch.zeros([num_steps],device=device)
    n_step_returns = [n_step_return_iterative(
            next_state_values=next_state_values,
            rewards=rewards,
            terminals=terminals,
            discount=discount,
            n = n+1
    ) for n in range(num_steps)]
    for i in range(num_steps):
        for j in range(num_steps-i-1):
            output[i] += (1-lam)*lam**j*n_step_returns[j][i]
        output[i] += (lam**(num_steps-i-1))*n_step_returns[num_steps-i-1][-1]
    return output
