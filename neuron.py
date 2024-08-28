import torch
from spikingjelly.activation_based import neuron, surrogate

from typing import Callable, Optional

# Defining the IF-neuron-model
class SingleSpikeIFNode(neuron.BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s'):
        """
        Basically an Integrate-and-Fire (IF) neuron without a reset of its membrane potential
        that can only omit at most one spike.

        Parameters:
        - v_threshold: The threshold potential for spiking.
        - v_reset: The reset potential after a spike (not used in this model).
        - surrogate_function: The surrogate gradient function for backpropagation.
        - detach_reset: Whether to detach the reset from the computation graph.
        - step_mode: The mode of operation (e.g., 's' for single-step mode).
        """
        
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.spiked = None
    
    # To handle input current
    def neuronal_charge(self, x: torch.Tensor): 
        if self.spiked == None: 
            self.spiked = torch.zeros_like(self.v)

        # Keep integrating input current x
        self.v = self.v + x

    # To determine whether a neuron fires
    def neuronal_fire(self):
        spikes = self.surrogate_function((1 - self.spiked) * self.v - self.v_threshold)
        # Get the non-zero indices from spikes
        spike_indices = spikes.nonzero(as_tuple=True)

        # Update self.spiked at the indices where spikes occurred
        self.spiked[spike_indices] = 1
        
        # Return the spike tensor indicating which neurons fired
        return spikes 
    
    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        # no membrane potential reset
        v = v 
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        # no membrane potential reset
        v = v 
        return v