import torch
from jaxtyping import Float, Bool

def v_trace_return(
    log_action_probs: Float[torch.Tensor, 'num_steps batch_shape'],
    old_log_action_probs: Float[torch.Tensor, 'num_steps batch_shape'],
    state_values: Float[torch.Tensor, 'num_steps batch_shape'],
    next_state_values: Float[torch.Tensor, 'num_steps batch_shape'],
    rewards: Float[torch.Tensor, 'num_steps batch_shape'],
    terminals: Float[torch.Tensor, 'num_steps batch_shape'],
    discount: float,
    max_c: float,
    max_rho: float,
    lam: float = 1.0,
):
    """
    V-trace return.
    See equation (1) and remark 1 in https://arxiv.org/pdf/1802.01561.pdf

    Args:
        log_action_probs: A tensor of length num_steps. The element at index t is $\\log\\pi(a_t|s_t)$.
        old_log_action_probs: A tensor of length num_steps. The element at index t is $\\log\\mu(a_t|s_t)$, where $\\mu$ is the policy that was originally used to sample action $a_t$.
        state_values: A tensor of length num_steps. The element at index t is $V(s_t)$.
        next_state_values: A tensor of length num_steps. The element at index t is $V(s_{t+1})$.
        rewards: A tensor of length num_steps. The element at index t is $r_t$.
        terminals: A tensor of length num_steps. The element at index t is $T(s_{t+1})$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discount: The discount factor.
        max_c: The maximum value of the importance sampling ratio $c$. Denoted by $\\bar c$ in the paper.
        max_rho: The maximum value of the importance sampling ratio $\rho$. Denoted by $\\bar \\rho$ in the paper.
        lam: Lambda parameter of TD(lambda). See remark 2 in the paper for details.
    """
    num_steps = log_action_probs.shape[0]
    shape = log_action_probs.shape[1:]
    device = log_action_probs.device

    rho = (log_action_probs - old_log_action_probs).exp().clamp(max=max_rho)
    c = lam*(log_action_probs - old_log_action_probs).exp().clamp(max=max_c)
    delta_v = rho * (rewards + discount * next_state_values * terminals.logical_not() - state_values)
    v_trace = torch.empty([num_steps+1, *shape], device=device)
    v_trace[-1, ...] = next_state_values[-1, ...]
    for i in range(num_steps-1, -1, -1):
        v_trace[i,...] = state_values[i,...] \
                + delta_v[i,...] \
                + discount \
                    * c[i,...] \
                    * (v_trace[i+1,...] - next_state_values[i,...]) \
                    * terminals[i,...].logical_not()
    return v_trace[:-1, ...]
