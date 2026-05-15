import pytest
import torch

from frankenstein.value.v_trace import v_trace_return, v_trace_return_iterative
from frankenstein.value.lam import lambda_return_iterative


def test_importance_sampling_ratios_clipped_1_step():
    """
    IS Ratio = 0.5 / 0.1 = 5, but should be clipped to 1.5, so the V-trace return should be the same as the lambda return with IS Ratio = 1.5.
    """
    output = v_trace_return_iterative(
        log_action_probs = torch.tensor([0.5]).log(),
        old_log_action_probs = torch.tensor([0.1]).log(),
        state_values = torch.tensor([2.0]),
        next_state_values = torch.tensor([3.0]),
        rewards = torch.tensor([1.0]),
        terminated = torch.tensor([False]),
        truncated = torch.tensor([False]),
        discount = 0.9,
        max_c = 1.5,
        max_rho = 1.5,
        lam = 1.0,
    )
    r"""
    \[v_s = V(x_s) + \sum_{t=s}^{s+n-1}\gamma^{t-s}\left(\prod_{i=s}^{t-1}c_i\right)\delta_tV\]
    Let $s = 0$
    Let $n = 1$
    \[
    \begin{align*}
        v_0
        &= V(x_0) + \sum_{t=0}^{0+1-1}\gamma^{t-0}\left(\prod_{i=0}^{t-1}c_i\right)\delta_tV \\
        &= V(x_0) + \sum_{t=0}^{0}\gamma^{t}\left(\prod_{i=0}^{t-1}c_i\right)\delta_tV \\
        &= V(x_0) + \gamma^{0}\left(\prod_{i=0}^{0-1}c_i\right)\delta_0V \\
        &= V(x_0) + \cancel{\gamma^{0}}\cancel{\left(\prod_{i=0}^{-1}c_i\right)}{}\delta_0V \\
        &= V(x_0) + \delta_0V \\
        &= V(x_0) + \rho_0(r_0 + \gamma V(x_1) - V(x_0)) \\
    \end{align*}
    \]
    """
    v0 = 2.0 + 1.5 * (1.0 + 0.9 * 3.0 - 2.0)  # IS Ratio clipped to 1.5
    #   └─┬─┘ └─┬─┘  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
    #     │     │      │     │     │     └─ V(x_0) = 2.0
    #     │     │      │     │     └─ V(x_1) = 3.0
    #     │     │      │     └─ \gamma = 0.9
    #     │     │      └─ r_0 = 1.0
    #     │     └─ IS Ratio = 1.5 (clipped from 5)
    #     └─ V(x_0) = 2.0
    assert output[0] == pytest.approx(v0)


def test_importance_sampling_ratios_clipped_2_steps():
    """
    IS Ratio = 0.5 / 0.1 = 5, but should be clipped to 1.5, so the V-trace return should be the same as the lambda return with IS Ratio = 1.5.
    """
    output = v_trace_return_iterative(
        log_action_probs = torch.tensor([0.5, 0.5]).log(),
        old_log_action_probs = torch.tensor([0.1, 0.1]).log(),
        state_values = torch.tensor([2.0, 3.0]),
        next_state_values = torch.tensor([3.0, 4.0]),
        rewards = torch.tensor([1.0, 1.0]),
        terminated = torch.tensor([False, False]),
        truncated = torch.tensor([False, False]),
        discount = 0.9,
        max_c = 1.5,
        max_rho = 1.5,
        lam = 1.0,
    )
    r"""
    \[v_s = V(x_s) + \sum_{t=s}^{s+n-1}\gamma^{t-s}\left(\prod_{i=s}^{t-1}c_i\right)\delta_tV\]
    Let $s = 0$
    Let $n = 2$
    \[
    \begin{align*}
        v_0
        &= V(x_s) + \sum_{t=s}^{s+n-1}\gamma^{t-s}\left(\prod_{i=s}^{t-1}c_i\right)\delta_tV \\
        &= V(x_0) + \sum_{t=0}^{1}\gamma^{t}\left(\prod_{i=0}^{t-1}c_i\right)\delta_tV \\
        &= V(x_0) + \cancel{\gamma^{0}}\cancel{\left(\prod_{i=0}^{-1}c_i\right)}\delta_0V + \gamma^{1}\left(\prod_{i=0}^{1-1}c_i\right)\delta_1V \\
        &= V(x_0) + \delta_0V + \gamma c_0\delta_1V \\
    \end{align*}
    \]
    """
    v0 = 2.0 + 1.5 * (1.0 + 0.9 * 3.0 - 2.0) + 0.9 * 1.5 * 1.5 * (1.0 + 0.9 * 4.0 - 3.0)  # IS Ratio clipped to 1.5
    assert output[0] == pytest.approx(v0)


def test_importance_sampling_ratios_unclipped_c_2_steps():
    """
    c is clipped, rho is not.
    """
    output = v_trace_return_iterative(
        log_action_probs = torch.tensor([0.5, 0.5]).log(),
        old_log_action_probs = torch.tensor([0.1, 0.1]).log(),
        state_values = torch.tensor([2.0, 3.0]),
        next_state_values = torch.tensor([3.0, 4.0]),
        rewards = torch.tensor([1.0, 1.0]),
        terminated = torch.tensor([False, False]),
        truncated = torch.tensor([False, False]),
        discount = 0.9,
        max_c = 100,
        max_rho = 1.5,
        lam = 1.0,
    )
    r"""
    \[v_s = V(x_s) + \sum_{t=s}^{s+n-1}\gamma^{t-s}\left(\prod_{i=s}^{t-1}c_i\right)\delta_tV\]
    Let $s = 0$
    Let $n = 2$
    \[
    \begin{align*}
        v_0
        &= V(x_s) + \sum_{t=s}^{s+n-1}\gamma^{t-s}\left(\prod_{i=s}^{t-1}c_i\right)\delta_tV \\
        &= V(x_0) + \sum_{t=0}^{1}\gamma^{t}\left(\prod_{i=0}^{t-1}c_i\right)\delta_tV \\
        &= V(x_0) + \cancel{\gamma^{0}}\cancel{\left(\prod_{i=0}^{-1}c_i\right)}\delta_0V + \gamma^{1}\left(\prod_{i=0}^{1-1}c_i\right)\delta_1V \\
        &= V(x_0) + \delta_0V + \gamma c_0\delta_1V \\
    \end{align*}
    \]
    """
    v0 = 2.0 + 1.5 * (1.0 + 0.9 * 3.0 - 2.0) + 0.9 * 5 * 1.5 * (1.0 + 0.9 * 4.0 - 3.0)  # IS Ratio clipped to 1.5
    assert output[0] == pytest.approx(v0)


def test_importance_sampling_ratios_unclipped_2_steps():
    """
    Neither c nor rho are clipped.
    """
    output = v_trace_return_iterative(
        log_action_probs = torch.tensor([0.5, 0.5]).log(),
        old_log_action_probs = torch.tensor([0.1, 0.1]).log(),
        state_values = torch.tensor([2.0, 3.0]),
        next_state_values = torch.tensor([3.0, 4.0]),
        rewards = torch.tensor([1.0, 1.0]),
        terminated = torch.tensor([False, False]),
        truncated = torch.tensor([False, False]),
        discount = 0.9,
        max_c = 100,
        max_rho = 100,
        lam = 1.0,
    )
    r"""
    \[v_s = V(x_s) + \sum_{t=s}^{s+n-1}\gamma^{t-s}\left(\prod_{i=s}^{t-1}c_i\right)\delta_tV\]
    Let $s = 0$
    Let $n = 2$
    \[
    \begin{align*}
        v_0
        &= V(x_s) + \sum_{t=s}^{s+n-1}\gamma^{t-s}\left(\prod_{i=s}^{t-1}c_i\right)\delta_tV \\
        &= V(x_0) + \sum_{t=0}^{1}\gamma^{t}\left(\prod_{i=0}^{t-1}c_i\right)\delta_tV \\
        &= V(x_0) + \cancel{\gamma^{0}}\cancel{\left(\prod_{i=0}^{-1}c_i\right)}\delta_0V + \gamma^{1}\left(\prod_{i=0}^{1-1}c_i\right)\delta_1V \\
        &= V(x_0) + \delta_0V + \gamma c_0\delta_1V \\
    \end{align*}
    \]
    """
    v0 = 2.0 + 5 * (1.0 + 0.9 * 3.0 - 2.0) + 0.9 * 5 * 5 * (1.0 + 0.9 * 4.0 - 3.0)  # IS Ratio clipped to 1.5
    assert output[0] == pytest.approx(v0)


def test_importance_sampling_ratios_unclipped_2_steps_at_end_of_episode():
    """
    Same as test_importance_sampling_ratios_unclipped_2_steps, but we add a few steps at the beginning to make sure that the function computes the same V-trace.
    """
    output = v_trace_return_iterative(
        log_action_probs = torch.tensor([1, 1, 1, 0.5, 0.5]).log(),
        old_log_action_probs = torch.tensor([1, 1, 1, 0.1, 0.1]).log(),
        state_values = torch.tensor([1, 1, 1, 2.0, 3.0]),
        next_state_values = torch.tensor([1, 1, 1, 3.0, 4.0]),
        rewards = torch.tensor([1, 1, 1, 1.0, 1.0]),
        terminated = torch.tensor([False, False, False, False, False]),
        truncated = torch.tensor([False, False, False, False, False]),
        discount = 0.9,
        max_c = 100,
        max_rho = 100,
        lam = 1.0,
    )
    v3 = 2.0 + 5 * (1.0 + 0.9 * 3.0 - 2.0) + 0.9 * 5 * 5 * (1.0 + 0.9 * 4.0 - 3.0)  # IS Ratio clipped to 1.5
    assert output[3] == pytest.approx(v3)


# Comparing to lambda return when IS Ratios are 1 (so that the V-trace return should be the same as the lambda return)


@pytest.mark.parametrize("n", [1, 2, 30])
def test_iterative_matches_lambda_return(n):
    """
    If log_action_probs and old_log_action_probs are the same (so that the importance sampling ratios are 1), then the V-trace return should be the same as the lambda return.
    """
    lam = 0.9
    discount = 0.7

    log_action_probs = torch.rand(n)

    state_values = torch.rand(n+1)
    rewards = torch.rand(n)

    terminated = torch.rand(n) < 0.2
    truncated = torch.rand(n) < 0.2

    output1 = v_trace_return_iterative(
        log_action_probs = log_action_probs,
        old_log_action_probs = log_action_probs,
        state_values = state_values[:n],
        next_state_values = state_values[1:],
        rewards = rewards,
        terminated = terminated,
        truncated = truncated,
        discount = discount,
        max_c = 1.5,
        max_rho = 1.5,
        lam = lam,
    )
    output2 = lambda_return_iterative(
        next_state_values = state_values[1:],
        rewards = rewards,
        truncated = truncated,
        terminated = terminated,
        discount = discount,
        lam = lam,
    )
    assert torch.allclose(output1, output2)


def test_matches_lambda_return():
    """
    If log_action_probs and old_log_action_probs are the same (so that the importance sampling ratios are 1), then the V-trace return should be the same as the lambda return.
    """
    n = 30
    lam = 0.9

    log_action_probs = torch.rand(n)
    state_values = torch.rand(n+1)
    rewards = torch.rand(n)
    terminated = torch.rand(n) < 0.2
    truncated = torch.rand(n) < 0.2

    output1 = v_trace_return(
        log_action_probs = log_action_probs,
        old_log_action_probs = log_action_probs,
        state_values = state_values[:n],
        next_state_values = state_values[1:],
        rewards = rewards,
        terminated = terminated,
        truncated = truncated,
        discount = 0.9,
        max_c = 1.5,
        max_rho = 1.5,
        lam = lam,
    )
    output2 = lambda_return_iterative(
        next_state_values = state_values[1:],
        rewards = rewards,
        truncated = truncated,
        terminated = terminated,
        discount = 0.9,
        lam = lam,
    )
    assert torch.allclose(output1, output2)


@pytest.mark.parametrize("n", [1, 2, 30])
def test_batch_matches_non_batched_1D(n):
    batch_size = 10

    log_action_probs = torch.rand(n, batch_size)
    old_log_action_probs = torch.rand(n, batch_size)
    state_values = torch.rand(n+1, batch_size)
    rewards = torch.rand(n, batch_size)
    terminated = torch.rand(n, batch_size) > 0.2
    truncated = torch.rand(n, batch_size) > 0.2

    output_batch = v_trace_return(
        log_action_probs = log_action_probs,
        old_log_action_probs = old_log_action_probs,
        state_values = state_values[:n,:],
        next_state_values = state_values[1:,:],
        rewards = rewards,
        truncated = truncated,
        terminated = terminated,
        discount = 0.9,
        max_c = 0.5,
        max_rho = 0.6,
    )
    for i in range(batch_size):
        output = v_trace_return_iterative(
            log_action_probs = log_action_probs[:,i],
            old_log_action_probs = old_log_action_probs[:,i],
            state_values = state_values[:n,i],
            next_state_values = state_values[1:,i],
            rewards = rewards[:,i],
            truncated = truncated[:,i],
            terminated = terminated[:,i],
            discount = 0.9,
            max_c = 0.5,
            max_rho = 0.6,
        )
        assert torch.allclose(output, output_batch[:,i])


def test_batch_matches_non_batched_3D():
    n = 10
    batch_size_1 = 3
    batch_size_2 = 4
    batch_size_3 = 5

    log_action_probs = torch.rand(n, batch_size_1, batch_size_2, batch_size_3)
    old_log_action_probs = torch.rand(n, batch_size_1, batch_size_2, batch_size_3)
    state_values = torch.rand(n+1, batch_size_1, batch_size_2, batch_size_3)
    rewards = torch.rand(n, batch_size_1, batch_size_2, batch_size_3)
    truncated = torch.rand(n, batch_size_1, batch_size_2, batch_size_3) > 0.2
    terminated = torch.rand(n, batch_size_1, batch_size_2, batch_size_3) > 0.2

    output_batch = v_trace_return(
        log_action_probs = log_action_probs,
        old_log_action_probs = old_log_action_probs,
        state_values = state_values[:n, :, :, :],
        next_state_values = state_values[1:, :, :, :],
        rewards = rewards,
        truncated = truncated,
        terminated = terminated,
        discount = 0.9,
        max_c = 0.5,
        max_rho = 0.6,
    )
    for i in range(batch_size_1):
        for j in range(batch_size_2):
            for k in range(batch_size_3):
                output = v_trace_return_iterative(
                    log_action_probs = log_action_probs[:,i,j,k],
                    old_log_action_probs = old_log_action_probs[:,i,j,k],
                    state_values = state_values[:n,i,j,k],
                    next_state_values = state_values[1:,i,j,k],
                    rewards = rewards[:,i,j,k],
                    truncated = truncated[:,i,j,k],
                    terminated = terminated[:,i,j,k],
                    discount = 0.9,
                    max_c = 0.5,
                    max_rho = 0.6,
                )
                assert torch.allclose(output, output_batch[:,i,j,k])
