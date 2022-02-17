from torchtyping import TensorType


def compute_expected_state_value(
        state_option_value: TensorType['num_options'],
        termination_prob: float,
        option: int,
        eps: float,
        deliberation_cost: float = 0):
    """
    Compute the expected value for the state $s_{t+1}$.

    Source: [When Waiting is not an Option : Learning Options with a Deliberation Cost](https://arxiv.org/pdf/1709.04571.pdf)

    Equation (8) says

    $$Q^c_\\theta(s_{t},o_{t}) = \\sum_{a_{t}} \\pi_\\theta(a_{t}|s_{t},o_{t})\\left(r(s_{t},a_{t})+\\gamma\\sum_{s_{t+1}}P(s_{t+1}|s_{t},a_{t}){\\color{red}\\left[Q^c_\\theta(s_{t+1},o_{t})-\\beta_\\theta(s_{t+1},o_{t})(A^c_\\theta(s_{t+1},o_{t})+\\eta)\\right]}\\right)$$

    `compute_expected_state_value` returns the portion highlighted in red.

    We can expand $A(s,o) = Q(s,o)-V(s)$ to get


    $$\\begin{align}
    &Q^c_\\theta(s_{t+1},o_{t})-\\beta_\\theta(s_{t+1},o_{t})(A^c_\\theta(s_{t+1},o_{t})+\\eta) \\\\\\\\
    &= Q^c_\\theta(s_{t+1},o_{t})-\\beta_\\theta(s_{t+1},o_{t})(Q^c_\\theta(s_{t+1},o_{t})-V^c_\\theta(s_{t+1})+\\eta) \\\\\\\\
    &= (1-\\beta_\\theta(s_{t+1},o_{t})) Q^c_\\theta(s_{t+1},o_{t})+\\beta_\\theta(s_{t+1},o_{t})(V^c_\\theta(s_{t+1})-\\eta) \\\\\\\\
    \\end{align}$$

    Args:
        state_option_value: The state-option value for all options at state $s_{t+1}$.
        termination_prob: The probability of terminating option $o_t$ at state $s_{t+1}$ and acting according to a different option.
            This termination occurs before an action is taken at state $s_{t+1}$.
        option: The option $o_t$ that was chosen to execute at state $s_{t}$. This same option will be used at time step
            $t+1$ unless terminated (with probability $\\beta$)
        eps: The probability of choosing a random option if the current option is terminated.
        deliberation_cost: The cost $\\eta$ of choosing a new option.
    """
    current_option_val = state_option_value[option]
    best_option_val = (1 - eps) * state_option_value.max() \
                         + eps  * state_option_value.mean()
    return (1 - termination_prob) * current_option_val \
              + termination_prob  * (best_option_val - deliberation_cost)


def compute_expected_state_value_batch(
        state_option_value: TensorType['batch_size','num_options'],
        termination_prob: TensorType['batch_size'],
        option: TensorType['batch_size'],
        eps: float,
        deliberation_cost: float = 0):
    """
    Compute the expected value for each state $s_{t+1}$ in the batch.
    See `compute_expected_state_value` for details.

    Args:
        state_option_value: The state-option value for all options at state $s_{t+1}$.
        termination_prob: The probability of terminating option $o_t$ at state $s_{t+1}$ and acting according to a different option.
            This termination occurs before an action is taken at state $s_{t+1}$.
        option: The option $o_t$ that was chosen to execute at state $s_{t}$. This same option will be used at time step
            $t+1$ unless terminated (with probability $\\beta$)
        eps: The probability of choosing a random option if the current option is terminated.
        deliberation_cost: The cost $\\eta$ of choosing a new option.
    """
    batch_size = state_option_value.shape[0]
    current_option_val = state_option_value[range(batch_size), option]
    best_option_val = (1 - eps) * state_option_value.max(1)[0] \
                         + eps  * state_option_value.mean(1)
    return (1 - termination_prob) * current_option_val \
              + termination_prob  * (best_option_val - deliberation_cost)


def gather_option_action(
        data: TensorType['seq_len','num_options','num_actions'],
        option: TensorType['seq_len'],
        action: TensorType['seq_len']) -> TensorType['seq_len']:
    seq_len = data.shape[0]
    return data[range(seq_len), option, action].squeeze(0)


def gather_option_action_batch(
        data: TensorType['seq_len','batch_size','num_options','num_actions'],
        option: TensorType['seq_len','batch_size'],
        action: TensorType['seq_len','batch_size']) -> TensorType['seq_len','batch_size']:
    # https://discuss.pytorch.org/t/similar-to-torch-gather-over-two-dimensions/118827/3
    seq_len, batch_size, num_options, num_actions = data.shape
    data = data.contiguous().view(seq_len, batch_size, num_options * num_actions)
    indices = option*num_actions + action
    return data.gather(-1,indices.unsqueeze(-1)).squeeze(-1)


def compute_termination_loss(
        termination_prob: TensorType['seq_len'],
        option_values_current: TensorType['seq_len'],
        option_values_max: TensorType['seq_len'],
        termination_reg: float = 0,
        deliberation_cost: float = 0):
    """
    Option termination gradient.

    Given a sequence of length n, we observe states/options/actions/rewards r_0,s_0,o_0,a_0,r_1,s_1,o_1,a_1,r_2,s_2,...,r_{n-1},s_{n-1},o_{n-1},a_{n-1}.

    Args:
        termination_prob: A 1D tensor of n elements where the element at index i is the predicted probability of terminating option o_{i-1} at state s_i.
        option_values_current: A 1D tensor of n elements. The element at index i is the value of option o_{i-1} at state s_i.
        option_values_max: A 1D tensor of n elements. The element at index i is the largest option value over all options at state s_i.
        termination_reg (float): A regularization constant to ensure that termination probabilities do not all converge on 1.
    """
    advantage = option_values_current-option_values_max+termination_reg
    advantage = advantage.detach()
    loss = (termination_prob*(advantage+deliberation_cost)).mean(0)
    return loss


def compute_termination_loss_batch(
        termination_prob,
        option_values_current,
        option_values_max,
        termination_reg,
        deliberation_cost):
    advantage = option_values_current-option_values_max+termination_reg
    advantage = advantage.detach()
    loss = (termination_prob*(advantage+deliberation_cost)).mean(1).mean(0)
    return loss
