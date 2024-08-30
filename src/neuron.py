import torch
from spikingjelly.activation_based import neuron, surrogate

from typing import Callable, Optional

class SingleSpikeIFNode(neuron.BaseNode):
    def __init__(self, v_threshold: float = 1.,
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 step_mode ='s'):
        """
        A modified Integrate-and-Fire (IF) neuron model without membrane potential reset.
        This neuron continuously integrates input current into its potential,
        emitting only one spike once its threshold is reached and not spiking
        again until it is completely reset for the next input presentation while
        still integrating input into the potential.
        Thus, it can emit at most one spike during an input presentation.

        Args:
            threshold (float): The threshold potential for spiking.
            surrogate_function (callable): The surrogate gradient function for backpropagation.
            step_mode (str): The mode of operation, 's' for single-step mode or 'm' for multi-step mode.
        """
        super().__init__(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
        self.spiked = None
    

    def neuronal_charge(self, x: torch.Tensor): 
        """
        Update the membrane potential of the neuron by integrating the input current.
    
        Args:
            x (torch.Tensor): Input current to be integrated.
        """
        # Initialize self.spiked to monitor neurons that have spiked
        # Initialization happens here because the shape depends on the input shape
        if self.spiked == None: 
            self.spiked = torch.zeros_like(self.v)

        # Keep integrating input current x
        self.v = self.v + x

    def neuronal_fire(self):
        """
        Compute neuron spikes based on the current membrane potential and update the spike record.
    
        Returns:
            torch.Tensor: A tensor indicating spike events.
        """
        # Check if a neuron that hasn't spiked before has now crossed the threshold
        spikes = self.surrogate_function((1 - self.spiked) * self.v - self.v_threshold)

        # Get the non-zero indices from spikes
        spike_indices = spikes.nonzero(as_tuple=True)

        # Update self.spiked at the indices where spikes occurred
        self.spiked[spike_indices] = 1
        
        return spikes 
    
    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        """
        Do not perform a hard reset of the membrane potential.
        This function is called automatically to perform a hard reset of the
        potential and is overwritten to not reset it for this neuron model.

        Args:
            v (torch.Tensor): The current membrane potential.
            spike (torch.Tensor): Tensor indicating spike events (unused here).
            v_reset (float): The reset value for the membrane potential (unused here).

        Returns:
            torch.Tensor: The (unchanged) membrane potential.
        """
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        """
        Do not perform a soft reset of the membrane potential.
        This function is called automatically to perform a soft reset of the
        potential and is overwritten to not reset it for this neuron model.

        Args:
            v (torch.Tensor): The current membrane potential.
            spike (torch.Tensor): Tensor indicating spike events (unused here).
            v_threshold (float): The threshold value for resetting (unused here).

        Returns:
            torch.Tensor: The (unchanged) membrane potential.
        """
        return v