import torch
from torchtyping import TensorType


def advantage_policy_gradient_loss(
    log_action_probs: TensorType['num_steps', float],
    terminals: TensorType['num_steps', bool],
    advantages: TensorType['num_steps', float],
) -> TensorType['num_steps', float]:
    """
    Advantage policy gradient.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        log_action_probs: A tensor containing $[\\log\\pi(a_0|s_0),\\log\\pi(a_1|s_1),\\cdots,\\log\\pi(a_{n-1}|s_{n-1})]$.
        terminals: A boolean tensor containing $[T(s_0),T(s_1),\\cdots,T(s_{n-1})]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        advantages: A tensor $[A^\\pi(s_0,a_0), A^\\pi(s_1,a_1),\\cdots, A^\\pi(s_{n-1},a_{n-1})]$ where $A^\\pi(s,a)=Q^\\pi(s,a)-V^\\pi(s)$ is the advantage of taking action $a$ at state $s$ over following policy $\\pi$.
    """
    return -log_action_probs*advantages*terminals.logical_not()


def clipped_advantage_policy_gradient_loss(
    log_action_probs: TensorType['batch_size', float],
    old_log_action_probs: TensorType['batch_size', float],
    terminals: TensorType['num_steps', bool],
    advantages: TensorType['batch_size', float],
    epsilon: float,
) -> TensorType['batch_size', float]:
    """
    Clipped advantage policy gradient.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        log_action_probs: A tensor containing $[\\log\\pi(a_0|s_0),\\log\\pi(a_1|s_1),\\cdots,\\log\\pi(a_{n-1}|s_{n-1})]$.
        terminals: A boolean tensor containing $[T(s_0),T(s_1),\\cdots,T(s_{n-1})]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        advantages: A tensor $[A^\\pi(s_0,a_0), A^\\pi(s_1,a_1),\\cdots, A^\\pi(s_{n-1},a_{n-1})]$ where $A^\\pi(s,a)=Q^\\pi(s,a)-V^\\pi(s)$ is the advantage of taking action $a$ at state $s$ over following policy $\\pi$.
    """
    ratio = torch.exp(log_action_probs - old_log_action_probs)
    return -torch.min(
        ratio * advantages,
        ratio.clip(1-epsilon, 1+epsilon) * advantages
    )*terminals.logical_not()

