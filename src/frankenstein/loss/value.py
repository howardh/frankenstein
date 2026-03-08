import torch

def clipped_value_loss(state_values, state_values_old, returns, clip_vf_loss):
    v_loss_unclipped = (state_values - returns) ** 2
    v_clipped = state_values_old + torch.clamp(
        state_values - state_values_old,
        -clip_vf_loss,
        clip_vf_loss,
    )
    v_loss_clipped = (v_clipped - returns) ** 2
    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    return 0.5 * v_loss_max
