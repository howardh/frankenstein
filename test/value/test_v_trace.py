import torch

from frankenstein.value.v_trace import v_trace_return
from frankenstein.value.lam import lambda_return_iterative


def test_matches_lambda_return():
    n = 30
    lam = 0.9

    log_action_probs = torch.rand(n)
    state_values = torch.rand(n+1)
    rewards = torch.rand(n)
    terminals = torch.rand(n) > 0.2

    output1 = v_trace_return(
        log_action_probs = log_action_probs,
        old_log_action_probs = log_action_probs,
        state_values = state_values[:n],
        next_state_values = state_values[1:],
        rewards = rewards,
        terminals = terminals,
        discount = 0.9,
        max_c = 1.5,
        max_rho = 1.5,
        lam = lam,
    )
    output2 = lambda_return_iterative(
        next_state_values = state_values[1:],
        rewards = rewards,
        terminals = terminals,
        discount = 0.9,
        lam = lam,
    )
    assert torch.allclose(output1, output2)


def test_batch_matches_non_batched_1D():
    n = 30
    batch_size = 10

    log_action_probs = torch.rand(n, batch_size)
    state_values = torch.rand(n+1, batch_size)
    rewards = torch.rand(n, batch_size)
    terminals = torch.rand(n, batch_size) > 0.2

    output_batch = v_trace_return(
        log_action_probs = log_action_probs,
        old_log_action_probs = log_action_probs,
        state_values = state_values[:n,:],
        next_state_values = state_values[1:,:],
        rewards = rewards,
        terminals = terminals,
        discount = 0.9,
        max_c = 0.5,
        max_rho = 0.6,
    )
    for i in range(batch_size):
        output = v_trace_return(
            log_action_probs = log_action_probs[:,i],
            old_log_action_probs = log_action_probs[:,i],
            state_values = state_values[:n,i],
            next_state_values = state_values[1:,i],
            rewards = rewards[:,i],
            terminals = terminals[:,i],
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
    state_values = torch.rand(n+1, batch_size_1, batch_size_2, batch_size_3)
    rewards = torch.rand(n, batch_size_1, batch_size_2, batch_size_3)
    terminals = torch.rand(n, batch_size_1, batch_size_2, batch_size_3) > 0.2

    output_batch = v_trace_return(
        log_action_probs = log_action_probs,
        old_log_action_probs = log_action_probs,
        state_values = state_values[:n, :, :, :],
        next_state_values = state_values[1:, :, :, :],
        rewards = rewards,
        terminals = terminals,
        discount = 0.9,
        max_c = 0.5,
        max_rho = 0.6,
    )
    for i in range(batch_size_1):
        for j in range(batch_size_2):
            for k in range(batch_size_3):
                output = v_trace_return(
                    log_action_probs = log_action_probs[:,i,j,k],
                    old_log_action_probs = log_action_probs[:,i,j,k],
                    state_values = state_values[:n,i,j,k],
                    next_state_values = state_values[1:,i,j,k],
                    rewards = rewards[:,i,j,k],
                    terminals = terminals[:,i,j,k],
                    discount = 0.9,
                    max_c = 0.5,
                    max_rho = 0.6,
                )
                assert torch.allclose(output, output_batch[:,i,j,k])
