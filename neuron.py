import torch
from spikingjelly.activation_based import neuron, surrogate

from typing import Callable, Optional

class SingleSpikeIFNode(neuron.BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s', N_out: int = 0):
        
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)

        self.spiked = torch.zeros(size=[N_out])

    def neuronal_charge(self, x: torch.Tensor): 
        # following comment refers to the commented code (4 lines below)
        # after a neuron spiked it's membrane potential stays 0
        # todo: discuss if the membrane potential should increase after a spike
        # but without being able to generate another action potential
        # self.v = (1 - self.spiked) * (self.v + x)

        # keep integrating input current x
        self.v = self.v + x

    def neuronal_fire(self):
        spikes = self.surrogate_function((1 - self.spiked) *self.v - self.v_threshold)
        self.spiked[spikes.nonzero()] = 1
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