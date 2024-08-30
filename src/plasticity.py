import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter

def f_weight(x):
    return torch.clamp(x, -1, 1.)

torch.manual_seed(0)

def get_k_winners(potentials: torch.Tensor, spikes: torch.Tensor, 
                  kwta=3, inhibition_radius=0):
    """
    Select the top-k winners based on spike timing and potentials with optional lateral inhibition.

    Args:
        potentials (torch.Tensor): Membrane potentials of shape (T, N, C, h, w).
        spikes (torch.Tensor): Spike events of shape (T, N, C, h, w).
        kwta (int, optional): Number of winners to select. Default is 3.
        inhibition_radius (int, optional): Radius for lateral inhibition. Default is 0.

    Returns:
        torch.Tensor: Indices of the winning neurons with shape (N, kwta, 3).
    """
    # Get shape size
    T, N, C, h, w = spikes.shape

    # Element-wise multiplication to get spike potentials (only non-zero where the neuron spiked)
    spike_potentials = potentials * spikes

    # Maximum potential used for rewarding early spikes
    maximum_potential = spike_potentials.max()

    # Reward early spikes by adding a bonus proportional to the remaining time steps, 
    # scaled by the maximum potential.
    time_steps = torch.arange(T-1, -1, -1).view(T, 1, 1, 1, 1).expand(-1, N, C, h, w) + 1
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
    else: # If inhibition is applied
        winners = []

        # Sum over the time dimension to aggregate potentials across all time steps, resulting in a tensor of shape [N, C, h, w]
        total = torch.sum(total, dim=0)

        for _ in range(kwta):
            # Reshape total_potential to (N, C * h * w, T) for top-k selection
            total_potential_flat = total.view(N, C * h * w)

            top_index = torch.argmax(total_potential_flat, dim=1)
        
            # Convert the flat indices to (C, h, w) coordinates
            c_index = (top_index // (h * w)).long()  # Channel index
            rem_index = top_index % (h * w)          # Remaining indices after channel removal
            y_index = (rem_index // w).long()        # Row index
            x_index = (rem_index % w).long()         # Column index

            # Stack the indices to get output shape (N, kwta, 3)
            winners.append(torch.stack([c_index, y_index, x_index], dim=1))  # Shape: (N, kwta, 3)

            # Inhibition around the winner to prevent nearby neurons from being selected
            for i in range(N):
                row_min = max(0, y_index[i] - inhibition_radius)
                row_max = min(h, y_index[i] + inhibition_radius + 1)
                col_min = max(0, x_index[i] - inhibition_radius)
                col_max = min(w, x_index[i] + inhibition_radius + 1)

                total[i, :, row_min:row_max, col_min:col_max] = 0
        
        winners = torch.stack(winners, dim=1)

    return winners

class STDP(nn.Module):
    def __init__(self, synapse, learning_rate, use_stabilizer = True, lower_bound = 0, upper_bound = 1):
        """
        Initialize the STDP (Spike-Timing-Dependent Plasticity) module.

        Args:
            synapse (nn.Module): The synapse module (e.g., convolutional layer) to which STDP will be applied.
            learning_rate (float or list of floats): Learning rates for LTP and LTD. If a list is provided,
                                                     it should be of length equal to the number of output channels.
            use_stabilizer (bool, optional): Whether to use a stabilizer for weight updates. Default is True.
            lower_bound (float, optional): Lower bound for the weight stabilizer. Default is 0.
            upper_bound (float, optional): Upper bound for the weight stabilizer. Default is 1.
        """
        super(STDP, self).__init__()
        self.synapse = synapse

        # Convert learning_rate to a list of parameters if it's not already a list
        if isinstance(learning_rate, list):
            self.learning_rate = learning_rate
        else:
            self.learning_rate = [learning_rate] * synapse.out_channels

        # Create parameters for each output channel
        for i in range(synapse.out_channels):
            self.learning_rate[i] = (Parameter(torch.tensor([self.learning_rate[i][0]])),
                            Parameter(torch.tensor([self.learning_rate[i][1]])))
            self.register_parameter('ltp_' + str(i), self.learning_rate[i][0])
            self.register_parameter('ltd_' + str(i), self.learning_rate[i][1])
            self.learning_rate[i][0].requires_grad_(False)
            self.learning_rate[i][1].requires_grad_(False)

        # Set stabilizer usage and bounds for weight updates
        self.use_stabilizer = use_stabilizer
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_spike_order(self, spikes_pre, spikes_post, winners):
        """
        Compute the spike order between pre- and post-synaptic neurons.

        Args:
            spikes_pre (torch.Tensor): Tensor of pre-synaptic spikes with shape [T, N_pre, C_pre, h_pre, w_pre].
            spikes_post (torch.Tensor): Tensor of post-synaptic spikes with shape [T, N_post, C_post, h_post, w_post].
            winners (torch.Tensor): Tensor of the k-Winner-take-all winners with shape [N, kwta, 3] containing the indices of the winners.

        Returns:
            torch.Tensor: Tensor indicating whether each pre-synaptic spike is earlier than the corresponding post-synaptic spike for each winner.
                        The tensor has shape [N, kwta, C_pre, kernel_width, kernel_height], where each entry is a boolean:
                        - `True` if the pre-synaptic spike is earlier than the post-synaptic spike.
                        - `False` otherwise.
        """

        # get the dimensionality of spikes_pre and spikes_post
        T, N_pre, C_pre, h_pre, w_pre = spikes_pre.shape
        _, N_post, C_post, h_post, w_post = spikes_post.shape

        # create tensors with same dimensionality but entries are T to 0 (early spikes have higher value)
        pre_times = (torch.arange(T-1, -1, -1).view(T, 1, 1, 1, 1).expand(-1, N_pre, C_pre, h_pre, w_pre) + 1) * spikes_pre
        post_times = (torch.arange(T-1, -1, -1).view(T, 1, 1, 1, 1).expand(-1, N_post, C_post, h_post, w_post) + 1) * spikes_post

        # reduce time dimensionality
        pre_times = torch.sum(pre_times, dim=0)
        post_times = torch.sum(post_times, dim=0)

        # Create a result tensor to store the pairings for STDP updates
        result = torch.zeros((winners.size(0), winners.size(1), C_pre, self.synapse.kernel_size[0], self.synapse.kernel_size[1]), dtype=torch.bool, device=spikes_pre.device)
        
        # Determine the spike order for each winner neuron
        for i in range(winners.size(0)):
            for j in range(winners.size(1)):
                winner = winners[i, j]

                input_tensor = pre_times[i, :, winner[-2]:winner[-2]+self.synapse.kernel_size[0], 
                                         winner[-1]:winner[-1]+self.synapse.kernel_size[1]]
                output_tensor = torch.ones(*self.synapse.kernel_size) * post_times[i, winner[0], winner[1], winner[2]]

                result[i, j] = torch.ge(input_tensor, output_tensor)
        
        return result # return shape=[N, kwth, C_pre, kernel_width, kernel_height]
    
    def forward(self, spikes_pre, spikes_post, potentials, winners=None, kwta=1, inhibition_radius=0, reward_batch=-1):
        """
        Perform the STDP update based on pre- and post-synaptic spikes and potentials.

        Args:
            spikes_pre (torch.Tensor): Tensor of pre-synaptic spikes with shape [T, N_pre, C_pre, h_pre, w_pre].
            spikes_post (torch.Tensor): Tensor of post-synaptic spikes with shape [T, N_post, C_post, h_post, w_post].
            potentials (torch.Tensor): Tensor of neuron potentials with shape [N, C_post, h_post, w_post].
            winners (torch.Tensor, optional): Tensor of winning neurons with shape [N, kwta, 3]. If None, winners are computed.
            kwta (int, optional): Number of winners to select. Default is 1.
            inhibition_radius (int, optional): Radius of inhibition for k-winners. Default is 0.
            reward_batch (int, optional): Index of the batch to apply reward to (Only R-STDP). Default is -1 (apply to all batches for STDP).
        """
        if winners is None:
            winners = get_k_winners(potentials=potentials, spikes=spikes_post, kwta=kwta, inhibition_radius=inhibition_radius)    
        
        # Get spike order pairings for STDP
        pairings = self.get_spike_order(spikes_pre=spikes_pre, spikes_post=spikes_post, winners=winners)
        
        # Initialize the weight change tensor
        dw = torch.zeros_like(self.synapse.weight)
        if reward_batch == -1: 
            # STDP: Apply STDP updates to all batches
            for i in range(winners.size(0)): # Iterate over each  batch
                lr = torch.zeros_like(self.synapse.weight)
                for j in range(winners.size(1)): # Interate over each winner
                    if potentials[i, winners[i,j,0], winners[i,j,1], winners[i,j,2]] > 0:
                        feature_map = winners[i][j][0]
                        lr[feature_map] = torch.where(pairings[i][j], *(self.learning_rate[feature_map]))

                dw += lr * ((self.synapse.weight-self.lower_bound) * (self.upper_bound-self.synapse.weight) if self.use_stabilizer else 1)
        else: 
            # R-STDP: Apply updates only to the weights of the specified reward_batch
            lr = torch.zeros_like(self.synapse.weight)
            for j in range(winners.size(1)): # kwta
                if potentials[reward_batch, winners[reward_batch,j,0], winners[reward_batch,j,1], winners[reward_batch,j,2]] > 0:
                    feature_map = winners[reward_batch][j][0]
                    lr[feature_map] = torch.where(pairings[reward_batch][j], *(self.learning_rate[feature_map]))
                    
            dw += lr * ((self.synapse.weight-self.lower_bound) * (self.upper_bound-self.synapse.weight) if self.use_stabilizer else 1)
            
        if self.synapse.weight.grad is None:
            # If the synapse weight gradient is not initialized, set it to the negative of the computed weight changes
            self.synapse.weight.grad = -dw
        else:
            # If the gradient is already initialized, subtract the computed weight changes from it
            self.synapse.weight.grad = self.synapse.weight.grad - dw

    def update_learning_rate(self, feature, ap, an):
        """
        Updates learning rate for a specific feature map.

        Args:
            feature (int): The target feature.
            ap (float): LTP rate.
            an (float): LTD rate.
        """
        self.learning_rate[feature][0][0] = ap
        self.learning_rate[feature][1][0] = an

    def update_all_learning_rate(self, ap, an):
        """
        Updates learning rates of all the feature maps to a same value.

        Args:
            ap (float): LTP rate.
            an (float): LTD rate.
        """
        for feature in range(self.synapse.out_channels):
            self.learning_rate[feature][0][0] = ap
            self.learning_rate[feature][1][0] = an

        