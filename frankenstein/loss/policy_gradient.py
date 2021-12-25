import torch
from torchtyping import TensorType

from .state_value import n_step_value_iterative

def geneeralized_advantage_estimate(
        state_values : TensorType['num_steps',float],
        next_state_values : TensorType['num_steps',float],
        rewards : TensorType['num_steps',float],
        terminals : TensorType['num_steps',bool],
        discounts : TensorType['num_steps',float],
        gae_lambda : float,
    ) -> TensorType['num_steps',float]:
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
        discounts: A tensor $[\\gamma_1,\\gamma_2,...,\\gamma_n]$ where the 1-step value estimate of $s_0$ is $r_1+\\gamma_1 V(s_1)$.
        gae_lambda: 
    """
    device = state_values.device
    num_steps = state_values.shape[0]
    if num_steps == 0:
        return torch.tensor([],dtype=torch.float,device=device)
    adv = torch.zeros([num_steps+1],device=device)
    delta = rewards+discounts*next_state_values*(1-terminals)-state_values
    for i in reversed(range(num_steps)):
        adv[i] = delta[i]+(1-terminals[i])*gae_lambda*discounts[i]*adv[i+1]
    return adv[:-1]

def advantage_policy_gradient_loss(
        log_action_probs : TensorType['num_steps',float],
        state_values : TensorType['num_steps',float],
        next_state_values : TensorType['num_steps',float],
        rewards : TensorType['num_steps',float],
        terminals : TensorType['num_steps',bool],
        prev_terminals : TensorType['num_steps',bool],
        discounts : TensorType['num_steps',float],
    ) -> TensorType['num_steps',float]:
    """
    Advantage policy gradient.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        log_action_probs: A tensor containing $[\\log\\pi(a_0|s_0),\\log\\pi(a_1|s_1),\\cdots,\\log\\pi(a_{n-1}|s_{n-1})]$.
        state_values: A tensor containing $[V(s_0),V_(s_1),\\cdots,V(s_{n-1})]$ where $V(s)$ is the estimated value of state $s$. These values are used as the policy gradient baseline.
        next_state_values: A tensor containing $[V(s_1),V(s_2),\\cdots,V(s_n)]$ where $V(s)$ is the estimated value of state $s$. These values are used for bootstrapping the n-step state value estimates.
        rewards: A tensor containing $[r_1,r_2,\\cdots,r_n]$.
        terminals: A boolean tensor containing $[T(s_1),T(s_2),\\cdots,T(s_n)]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        prev_terminals: A boolean tensor containing $[T(s_0),T(s_1),\\cdots,T(s_{n-1})]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discounts: A tensor $[\\gamma_1,\\gamma_2,\\cdots,\\gamma_n]$ where the 1-step value estimate of $s_0$ is $r_1+\\gamma_1 V(s_1)$.
    """
    vt = n_step_value_iterative(next_state_values, rewards, terminals, discounts)
    return -log_action_probs*(state_values-vt)*(1-prev_terminals)

def clipped_advantage_policy_gradient_loss(
        log_action_probs : TensorType['num_steps',float],
        old_log_action_probs : TensorType['num_steps',float],
        state_values : TensorType['num_steps',float],
        next_state_values : TensorType['num_steps',float],
        rewards : TensorType['num_steps',float],
        terminals : TensorType['num_steps',bool],
        prev_terminals : TensorType['num_steps',bool],
        discounts : TensorType['num_steps',float],
        epsilon : float,
    ) -> TensorType['num_steps',float]:
    """
    Clipped advantage policy gradient (See equation 7 of https://arxiv.org/pdf/1707.06347.pdf).

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        log_action_probs: A tensor containing $[\\log\\pi(a_0|s_0),\\log\\pi(a_1|s_1),\\cdots,\\log\\pi(a_{n-1}|s_{n-1})]$.
        old_log_action_probs: A tensor containing $[\\log\\pi'(a_0|s_0),\\log\\pi'(a_1|s_1),\\cdots,\\log\\pi'(a_{n-1}|s_{n-1})]$, where $\\pi'$ is the policy used to sample actions $a_0,\\cdots,a_n$.
        state_values: A tensor containing $[V(s_0),V_(s_1),\\cdots,V(s_{n-1})]$ where $V(s)$ is the estimated value of state $s$. These values are used as the policy gradient baseline.
        next_state_values: A tensor containing $[V(s_1),V(s_2),\\cdots,V(s_n)]$ where $V(s)$ is the estimated value of state $s$. These values are used for bootstrapping the n-step state value estimates.
        rewards: A tensor containing $[r_1,r_2,\\cdots,r_n]$.
        terminals: A boolean tensor containing $[T(s_1),T(s_2),\\cdots,T(s_n)]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        prev_terminals: A boolean tensor containing $[T(s_0),T(s_1),\\cdots,T(s_{n-1})]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discounts: A tensor $[\\gamma_1,\\gamma_2,\\cdots,\\gamma_n]$ where the 1-step value estimate of $s_0$ is $r_1+\\gamma_1 V(s_1)$.
    """
    vt = n_step_value_iterative(next_state_values, rewards, terminals, discounts)
    ratio = torch.exp(log_action_probs-old_log_action_probs)
    advantage = (state_values-vt)*prev_terminals.logical_not()
    return -torch.min(
            ratio*advantage,
            ratio.clip(1-epsilon,1+epsilon)*advantage
    )
