import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing
from spikingjelly.activation_based import encoding, learning
import cv2
from torchvision import datasets, transforms

from typing import Callable, Optional


use_cupy = False

class OneSpikeIF(neuron.BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', N_out: int = 0, store_v_seq=True):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset,
                         step_mode, store_v_seq=store_v_seq)

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
        if self.spiked.shape != self.v.shape:
            self.spiked = torch.zeros_like(self.v)
            print("reset spiked", self.spiked.shape)
        
        spikes = self.surrogate_function((1 - self.spiked) * self.v - self.v_threshold)
        
        # Get indices where spikes are non-zero
        spike_indices = spikes.nonzero()
        
        # Print for debugging
        print("self.spiked shape:", self.spiked.shape)
        print("spike_indices shape:", spike_indices.shape)
        
        # Set spiked to 1 at the locations where spikes are non-zero
        self.spiked[tuple(spike_indices.T)] = 1

        return spikes


    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        # no membrane potential reset
        v = v
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor,
                       v_threshold: float):
        # no membrane potential reset
        v = v
        return v


class SpikeTransform:
    def __init__(self, timesteps=15):
        self.temporal_transform = encoding.LatencyEncoder(timesteps)
        self.count = 0
        self.timesteps = timesteps
    def __call__(self, image):
        image = torch.as_tensor(image).squeeze().numpy()
        b1 = cv2.GaussianBlur(image,ksize=(3,3),sigmaX=3/9,sigmaY=6/9)
        b2 = cv2.GaussianBlur(image,ksize=(3,3),sigmaX=6/9,sigmaY=3/9)
        image1 = b1 - b2
        image2 = b2 - b1

        b3 = cv2.GaussianBlur(image,ksize=(7,7),sigmaX=7/9,sigmaY=14/9)
        b4 = cv2.GaussianBlur(image,ksize=(7,7),sigmaX=14/9,sigmaY=7/9)
        image3 = b3 - b4
        image4 = b4 - b3

        b5 = cv2.GaussianBlur(image,ksize=(13,13),sigmaX=13/9,sigmaY=26/9)
        b6 = cv2.GaussianBlur(image,ksize=(13,13),sigmaX=26/9,sigmaY=13/9)
        image5 = b6 - b5
        image6 = b5 - b6

        # print("max val", np.max(image1), np.max(image2), np.max(image3), np.max(image4), np.max(image5), np.max(image6))

        image = torch.stack([torch.as_tensor(image1), torch.as_tensor(image2), torch.as_tensor(image3), torch.as_tensor(image4), torch.as_tensor(image5), torch.as_tensor(image6)])
        image = torchvision.transforms.functional.convert_image_dtype(image, dtype=torch.int8)
        # print("max val", torch.max(image))
        # ignore values below 50
        image = F.threshold(image, threshold=50, value=0)
        temporal_image = torch.zeros([self.timesteps]+list(image.shape))
        # intensity to latency
        for t in range(self.timesteps):
            temporal_image[t] = self.temporal_transform(image)
        return temporal_image


# calculate stdp updates

# get weights to change -> those going from the conv to the IF neurons

# implement custom stdp and rstdp
class STDP():
    # at the final step if pre neuron has not fired, treat it as though post-before-pre and reduce connection strength
    def __init__(self, alpha, beta, ar_p, ar_n, ap_p, ap_n, phi_r, phi_p, use_stabilizer=True, perform_weight_clipping=False, upper_weight_limit=1.0, lower_weight_limit=0.0):
        self.alpha = alpha
        self.beta = beta
        self.ar_p = ar_p
        self.ar_n = ar_n
        self.ap_p = ap_p
        self.ap_n = ap_n
        self.phi_r = phi_r
        self.phi_p = phi_p
        self.use_stabilizer = use_stabilizer
        self.perform_weight_clipping = perform_weight_clipping
        self.upper_weight_limit = upper_weight_limit
        self.lower_weight_limit = lower_weight_limit

    def get_spike_order(self, spikes_pre, spikes_post):
        # get the cumulative sum for both spike tensors on the time dimension
        pre = torch.cumsum(spikes_pre, dim=0)
        post = torch.cumsum(spikes_post, dim=0)

        # assumes at max one spike per neuron
        spike_order = torch.sum(pre - post, dim=0) # sum across time - assumes at most one spike per neuron
        # clamp to get rid of negative values from when post fired before pre
        spike_order = torch.clamp(spike_order, min=0, max=1)

        return spike_order


    def forward(self, spikes_pre, spikes_post, weights, eligible_neurons=None):
        # eligible neurons should be a mask of the same shape as weights specifying which neurons/weights should be changed through STDP

        # get k winners because only the corresponding weights need to be updated - [TODO] do to outside of the STDP to more easily be able to replace this
        # if eligible neurons is none all neurons are updated, otherwise only those specified are
        if eligible_neurons is None:
            eligible_neurons = torch.ones_like(weights)

        pre_before_post = self.get_spike_order(spikes_pre, spikes_post)
        delta_pre = self.alpha * self.phi_r * self.ar_p + self.beta * self.phi_p * self.ap_n
        delta_post = self.alpha * self.phi_r * self.ar_n + self.beta * self.phi_p * self.ap_p

        # return weight update (delta_w)
        weight_updates = torch.where(pre_before_post, delta_pre, delta_post)

        # multiply with stabilizer term
        if self.use_stabilizer:
            weight_updates *= (weights - self.lower_weight_limit) / (self.upper_weight_limit - weights)

        if self.perform_weight_clipping:
            weight_updates = torch.clamp(weight_updates, min=self.lower_weight_limit, max=self.upper_weight_limit)

        # mask so that only the winners are updated
        delta_w = torch.where(eligible_neurons, weight_updates, torch.zeros_like(weights))

        # change weights
        weights += delta_w

    def forward_rstdp(self, spikes_pre, spikes_post, weights, alpha, beta, phi_r, phi_p, eligible_neurons=None):
        # in this case alpha and beta need to be tensors of the same shape as weight
        # (because they can be different for the different images in a batch, but need to match the shape to work with torch.where)
        self.alpha = alpha
        self.beta = beta
        self.phi_r = phi_r
        self.phi_p = phi_p
        self.forward(spikes_pre, spikes_post, weights, eligible_neurons)

    def complete_forward(self):
        #
        pass


def get_k_winners(spikes, potentials, k, r):
    # expects unbatched input
    number_of_selected_winners = 0
    # set the potential of neurons that spiked to the maximum after the spike so that processing can be done with only the potentials
    spike_indices = torch.nonzero(spikes)
    max_potential = torch.max(torch.max(potentials), torch.as_tensor(1.0))
    for spike in spike_indices:
        potentials[spike[0]:, spike[1], spike[2], spike[3]] = max_potential
    # sum over time
    potentials_summed = torch.sum(potentials, 0)
    eligible_neurons = torch.ones_like(potentials_summed)
    print("summed poshls", potentials_summed.shape, potentials_summed)
    # list to store winners in
    winners = []
    # get k neurons that spiked first
    while number_of_selected_winners < k and torch.max(potentials_summed) > 0:
        # get the neuron with the highest potential (also corresponds to the first spike if neurons spiked)
        winner_index = (potentials_summed == torch.max(potentials_summed)).nonzero()[0]
        # print("potentials", potentials_summed)
        # print("winner_index", winner_index)
        winner_feature_map, winner_x, winner_y = tuple(winner_index.tolist())
        # inhibit surrounding neurons in all feature maps - use a mask for that -> set to true everywhere in the beginning and then set to false for inhibited neurons  #
        winners.append(winner_index)
        number_of_selected_winners += 1
        x_min = max(winner_x-r, 0)
        x_max = min(winner_x+r+1, eligible_neurons.shape[-1])
        y_min = max(winner_y-r, 0)
        y_max = min(winner_y+r+1, eligible_neurons.shape[-1])
        eligible_neurons[:, x_min:x_max, y_min:y_max] = 0
        eligible_neurons[winner_feature_map,:,:] = 0
        # multiply potentials with eligibility mask so that inhibited neurons cannot be chosen
        potentials_summed *= eligible_neurons
    return winners


def get_k_winners_batched(spikes, potentials, k, r):
    # Ensure spikes and potentials are tensors and permute them
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.permute(1, 0, 2, 3, 4)  # Assuming spikes are of shape [T, N, C, H, W]
    else:
        raise TypeError("Expected spikes to be a Tensor, but got a different type.")

    if isinstance(potentials, torch.Tensor):
        potentials = potentials.permute(1, 0, 2, 3, 4)  # Assuming potentials are of shape [T, N, C, H, W]
    else:
        raise TypeError("Expected potentials to be a Tensor, but got a different type.")

    all_winners = []

    # Iterate through batches
    for batch_index in range(spikes.shape[0]):
        winners = get_k_winners(spikes[batch_index], potentials[batch_index], k, r)
        print("winners", winners, "batch", batch_index)
        all_winners.append(winners)

    return all_winners


    # Iterate through batches
    for batch_index in range(spikes.shape[0]):
        winners = get_k_winners(spikes[batch_index], potentials[batch_index], k, r)
        print("winners", winners, "batch", batch_index)
        all_winners.append(winners)

    return all_winners


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            # use convolution layer from spikingjelly because it is already
            # wrapped to support both step modes and works better when saving models

            # S1 #0
            layer.Conv2d(6, 30, kernel_size=5, bias=False),
            OneSpikeIF(v_threshold=15.0, surrogate_function=surrogate.ATan(), store_v_seq=True),
            # C1  - pooling (first to spike / maximum potential)
            # spike-based
            layer.MaxPool2d(kernel_size=2, stride=2),
            nn.ConstantPad2d((1, 1, 1, 1), 0),

            # S2 #4
            layer.Conv2d(30, 250, kernel_size=3, bias=False),
            OneSpikeIF(v_threshold=10.0, surrogate_function=surrogate.ATan(), store_v_seq=True),
            # C2
            # spike-based
            layer.MaxPool2d(kernel_size=3, stride=3),

            nn.ConstantPad2d((2,2,2,2), 0),

            # S3 #8
            layer.Conv2d(250, 200, kernel_size=5, bias=False),
            OneSpikeIF(v_threshold=float('inf'),
                       surrogate_function=surrogate.ATan(), store_v_seq=True),
            # C3 - global pooling, neurons are preassigned to a digit
            # potential-based
            # layer.MaxPool2d(kernel_size=5, stride=0),
            # [TODO] max pooling probably doesn't work here out of the box
            # (because neurons don't spike and we want to pool the internal potential)
            #  → can use get_k_winners instead, just like in the SpykeTorch code
        )
        # initialize conv weights
        def custom_weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.8, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        self.net[0].apply(custom_weights_init)
        self.net[4].apply(custom_weights_init)
        self.net[5].apply(custom_weights_init)

        # spike storage
        self.spikes = {}

    def __len__(self):
        return len(self.net)


    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)

            # Keep a record of the spikes of each layer for training purposes
            if isinstance(layer, OneSpikeIF):
                if i == 1:
                    self.s1_spikes = x
                elif i == 4:
                    self.s2_spikes = x
                elif i == 8:
                    self.s3_spikes = x

        return x

def main():
    # load MNIST dataset, filter with DoG filters and perform intensity to latency encoding
    transform = transforms.Compose([transforms.ToTensor(), SpikeTransform(), ])

    train_dataset = datasets.MNIST(root='data', train=True, transform=transform,
                                   download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform,
                                  download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                              shuffle=False)


    net = Network()


    # use multi-step mode for faster training
    functional.set_step_mode(net, step_mode='m')

    # using the cupy back end can speed up training
    if use_cupy:
        functional.set_backend(net, backend='cupy')

    instances_stdp = (layer.Conv2d,)

    # stdp_learners = []
    #
    # for i in range(net.__len__()):
    #     if isinstance(net[i], instances_stdp):
    #         stdp_learners.append(
    #                 learning.STDPLearner(step_mode="m", synapse=net[i],
    #                                      sn=net[i + 1], tau_pre=1.0,
    #                                      tau_post=1.0))
    #
    # params_stdp = []
    # for m in net.modules():
    #     if isinstance(m, instances_stdp):
    #         for p in m.parameters():
    #             params_stdp.append(p)
    #
    # # train in a layer by layer fashion
    # optimizer = SGD(params_stdp, lr=0.004, momentum=0.)

    for epoch in range(1):
        for frame, label in train_loader:
            frame = frame.to('cpu:0')
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]

            y = net(frame)

            # Print output shape
            print("Output y shape:", y.shape)

            # Print spike information for each layer
            for key, spikes in net.spikes.items():
                print(f"Spikes for layer {key} shape:", spikes.shape)

            # Extract spikes for the specific layer you want to analyze
            spikes_s1 = net.spikes.get('s1', None)
            spikes_s2 = net.spikes.get('s2', None)
            spikes_s3 = net.spikes.get('s3', None)

            if spikes_s1 is not None:
                print("Spikes S1 shape:", spikes_s1.shape)
            if spikes_s2 is not None:
                print("Spikes S2 shape:", spikes_s2.shape)
            if spikes_s3 is not None:
                print("Spikes S3 shape:", spikes_s3.shape)

            # Example: using spikes from the S1 layer
            if spikes_s1 is not None:
                k_winners = get_k_winners_batched(spikes_s1, net.net[1].v_seq, k=4, r=0)
                print("Winners:", k_winners)

            # Compute loss and backpropagate
           # print("Output y shape:", y.shape)
            #print("Label shape:", label.shape)
            #loss = F.cross_entropy(y, label)
            #print("Loss:", loss.item())
            #loss.backward()


        # test
        with torch.no_grad():
            for frame, label in test_loader:
                frame = frame.to('cpu:0')
                frame = frame.transpose(0, 1)

# main()

def test():
    spikes = torch.zeros((3, 2, 4, 4))
    spikes[1, 0, 1, 2] = 1
    spikes[0, 1, 3, 3] = 1
    w = get_k_winners(spikes, spikes, k=4, r=0)
    print(w)

    spikes1 = torch.zeros((3, 2, 4, 4))
    spikes1[2, 0, 0, 0] = 1
    spikes1[0, 1, 2, 1] = 1
    w = get_k_winners(spikes1, spikes1, k=4, r=0)
    print(w)

    spikes = torch.stack((spikes, spikes1))
    print(spikes.shape)
    spikes = spikes.permute(1, 0, 2, 3, 4)
    print(spikes.shape)
    w = get_k_winners_batched(spikes, spikes, k=4, r=0)


    print(w)

test()
main()