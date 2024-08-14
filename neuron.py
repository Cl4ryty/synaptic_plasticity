import torch
from spikingjelly.activation_based import neuron, surrogate

from typing import Callable, Optional

class SingleSpikeIFNode(neuron.BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s'):
        """
        Basically an Integrate-and-Fire (IF) neuron without a reset of its membrane potential
        that can only omit at most one spike.
        """
        
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.spiked = None
    

    def neuronal_charge(self, x: torch.Tensor): 
        if self.spiked == None: 
            self.spiked = torch.zeros_like(self.v)
        # following comment refers to the commented code (4 lines below)
        # after a neuron spiked it's membrane potential stays 0
        # todo: discuss if the membrane potential should increase after a spike
        # but without being able to generate another action potential
        # self.v = (1 - self.spiked) * (self.v + x)

        # keep integrating input current x
        self.v = self.v + x

    def neuronal_fire(self):
        spikes = self.surrogate_function((1 - self.spiked) * self.v - self.v_threshold)
        # Get the non-zero indices from spikes
        spike_indices = spikes.nonzero(as_tuple=True)

        # Update self.spiked at the indices where spikes occurred
        self.spiked[spike_indices] = 1
        # if type(self.spiked) == int and self.spiked == 0:
        #     self.spiked = 1 if spikes.nonzero().numel() != 0 else 0
        # elif type(self.spiked) != int:
        
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