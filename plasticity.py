import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter

def f_weight(x):
    return torch.clamp(x, -1, 1.)

torch.manual_seed(0)

def get_k_winners(potentials, spikes, kwta=3, inhibition_radius=0):

    # todo: inhibition_radius is missing 

    T, N, C, h, w = spikes.shape

    spike_potentials = potentials * spikes

    maximum_potential = spike_potentials.max()

    # 'reward' early spikes by adding the maximum potential times the number of time steps that follow after a spike
    time_steps = torch.arange(T-1, -1, -1).view(T, 1, 1, 1, 1).expand(-1, N, C, h, w)
    early_reward = time_steps * spikes * maximum_potential

    total = spike_potentials + early_reward

    # Reshape total_potential to (N, C * h * w, T) for top-k selection
    total = torch.sum(total, dim=0)
    print(total.shape)
    total_potential_flat = total.view(N, C * h * w)

    # Get the top kwta values and indices
    _, topk_indices = torch.topk(total_potential_flat, kwta, dim=1, largest=True, sorted=False)
    
    # Convert flat indices to (C, h, w) coordinates
    N, _ = total_potential_flat.shape

    # Convert the flat indices to (C, h, w) coordinates
    c_indices = (topk_indices // (h * w)).long()  # Channel index
    rem_indices = topk_indices % (h * w)          # Remaining indices after channel removal
    y_indices = (rem_indices // w).long()         # Row index
    x_indices = (rem_indices % w).long()          # Column index

    # Stack the indices to get output shape (N, kwta, 3)
    winners = torch.stack([c_indices, y_indices, x_indices], dim=2)  # Shape: (N, kwta, 3)

    return winners