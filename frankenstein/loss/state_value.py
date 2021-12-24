import torch
from torchtyping import TensorType

def n_step_value_iterative(
        state_values : TensorType['num_steps',float],
        rewards : TensorType['num_steps',float],
        terminals : TensorType['num_steps',bool],
        discounts : TensorType['num_steps',float]
    ) -> TensorType['num_steps',float]:
    """
    Return the n-step value of each state. The output is computed in an interative manner, so this is less computationally efficient, but the code may be easier to understand.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        state_values: A tensor containing $[V(s_1),V_(s_2),...,V(s_n)]$ where $V(s)$ is the estimated value of state $s$.
        rewards: A tensor containing $[r_1,r_2,...,r_n]$.
        terminals: A boolean tensor containing $[T(s_1),T(s_2),...,T(s_n)]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discounts: A tensor $[\\gamma_1,\\gamma_2,...,\\gamma_n]$ where the 1-step value estimate of $s_0$ is $r_1+\\gamma_1 V(s_1)$.
    """
    device = state_values.device
    num_steps = state_values.shape[0]
    if num_steps == 0:
        return torch.tensor([],dtype=torch.float,device=device)
    vt = torch.zeros([num_steps+1],device=device)
    if not terminals[-1]:
        vt[-1] = state_values[-1].item()
    for i in reversed(range(len(state_values))):
        vt[i] = rewards[i]+terminals[i].logical_not()*discounts[i]*vt[i+1]
    return vt[:-1]

def n_step_value_iterative_batch(
        state_values : TensorType['num_steps','batch_size',float],
        rewards : TensorType['num_steps','batch_size',float],
        terminals : TensorType['num_steps','batch_size',bool],
        discounts : TensorType['num_steps','batch_size',float],
    ) -> TensorType['num_steps','batch_size',float]:
    """
    Return the n-step value of each state. The output is computed in an interative manner, so this is less computationally efficient, but the code may be easier to understand.

    Given a sequence of n transitions, we observe states/actions/rewards
    $$(r_0,s_0,a_0),(r_1,s_1,a_1),...,(r_n,s_n,a_n)$$
    We treat $r_{i+1}$ as the reward for taking action $a_i$ at state $s_i$.

    Args:
        state_values: A tensor containing $[V(s_1),V_(s_2),...,V(s_n)]$ where $V(s)$ is the estimated value of state $s$.
        rewards: A tensor containing $[r_1,r_2,...,r_n]$.
        terminals: A boolean tensor containing $[T(s_1),T(s_2),...,T(s_n)]$ where $T(s)$ is True if $s$ is a terminal state, and False otherwise.
        discounts: A tensor $[\\gamma_1,\\gamma_2,...,\\gamma_n]$ where the 1-step value estimate of $s_0$ is $r_1+\\gamma_1 V(s_1)$.
    """
    device = state_values.device
    num_steps = state_values.shape[0]
    if num_steps == 0:
        return torch.empty_like(state_values,dtype=torch.float,device=device)
    batch_size = state_values.shape[1]
    vt = torch.zeros([num_steps+1,batch_size],device=device)
    vt[-1,:] = terminals[-1,:].logical_not()*state_values[-1,:]
    for i in reversed(range(len(state_values))):
        vt[i,:] = rewards[i,:]+terminals[i,:].logical_not()*discounts[i,:]*vt[i+1,:]
    return vt[:-1,:]
