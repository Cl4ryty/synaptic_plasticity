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
   
    if inhibition_radius == 0: 
        # Reshape total_potential to (N, C * h * w, T) for top-k selection
        total = torch.sum(total, dim=0)
        total_potential_flat = total.view(N, C * h * w)

        # Get the top kwta values and indices
        _, topk_indices = torch.topk(total_potential_flat, kwta, dim=1, largest=True, sorted=False)

        # Convert the flat indices to (C, h, w) coordinates
        c_indices = (topk_indices // (h * w)).long()  # Channel index
        rem_indices = topk_indices % (h * w)          # Remaining indices after channel removal
        y_indices = (rem_indices // w).long()         # Row index
        x_indices = (rem_indices % w).long()          # Column index

        # Stack the indices to get output shape (N, kwta, 3)
        winners = torch.stack([c_indices, y_indices, x_indices], dim=2)  # Shape: (N, kwta, 3)
    else: 
        winners = []

        total = torch.sum(total, dim=0)

        for _ in range(kwta):
            # Reshape total_potential to (N, C * h * w, T) for top-k selection
            total_potential_flat = total.view(N, C * h * w)

            top_index = torch.argmax(total_potential_flat, dim=1)

             # Convert the flat indices to (C, h, w) coordinates
            c_index = (top_index // (h * w)).long()  # Channel index
            rem_index = top_index % (h * w)          # Remaining indices after channel removal
            y_index = (rem_index // w).long()         # Row index
            x_index = (rem_index % w).long()          # Column index

            # Stack the indices to get output shape (N, kwta, 3)
            winners.append(torch.stack([c_index, y_index, x_index], dim=1))  # Shape: (N, kwta, 3)

            for i in range(N):
                row_min = max(0, y_index[i] - inhibition_radius)
                row_max = min(h, y_index[i] + inhibition_radius + 1)
                col_min = max(0, x_index[i] - inhibition_radius)
                col_max = min(w, x_index[i] + inhibition_radius + 1)

                total[i, :, row_min:row_max, col_min:col_max] = 0
        
        winners = torch.stack(winners, dim=1)

    return winners