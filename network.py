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


use_cupy = True

class OneSpikeIF(neuron.BaseNode):
    def __init__(self, v_threshold=1., v_reset=0.,
                 surrogate_function=surrogate.Sigmoid(), detach_reset=False,
                 step_mode='s', backend='torch', store_v_seq=False):
        super().__init__(v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.Sigmoid(),
                         detach_reset=False, step_mode='s', backend='torch',
                         store_v_seq=False)
        self.has_spiked = False

    def neuronal_fire(self):
        if not self.has_spiked:
            self.has_spiked = True
            return self.surrogate_function(self.v - self.v_threshold)
        else:
            return 0  # do not spike again

    def reset(self):
        self.has_spiked = False
        super().reset()

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

        image = torch.stack([torch.as_tensor(image1), torch.as_tensor(image2), torch.as_tensor(image3), torch.as_tensor(image4), torch.as_tensor(image5), torch.as_tensor(image6)])
        # ignore values below 50
        image = F.threshold(image, threshold=50, value=0)
        temporal_image = torch.zeros([self.timesteps]+list(image.shape))
        # intensity to latency
        for t in range(self.timesteps):
            temporal_image[t] = self.temporal_transform(image)
        return temporal_image.sign().byte()






def f_weight(x):
    return torch.clamp(x, -1, 1.)


# calculate stdp updates

# get weights to change -> those going from the conv to the IF neurons

# implement custom stdp and rstdp
def STDP():
    # at the final step if pre neuron has not fired, treat it as though post-before-pre and reduce connection strength
    pass

def get_k_winners(spikes, neurons, k, r):
    number_of_selected_winners = 0
    select_based_on_potential = False
    # get the potentials of the neurons / get the spike times
    potentials = neurons.v
    # get k neurons that spiked first
    while number_of_selected_winners < k:
        if not select_based_on_potential:
            # select based on spikes
            # get first spikes
            # for each timestep
            # for each feature map
            # get spikes - if there are more than one select the neuron with the highest potential - mask so that inhibited neurons cannot be selected
            # inhibit surrounding neurons in all feature maps - use a mask for that -> set to true everywhere in the beginning and then set to false for inhibited neurons
            #
            pass
        else:
            # if no neurons spiked, select those with the highest potential
            pass



train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(),
                                   download=True)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True)
for image, t in train_loader:
    print("i", image.shape)
    print("t", t.shape)

    # st = SpikeTransform()
    # out = st(i)
    image = torch.as_tensor(image).squeeze().numpy()
    cv2.imshow('image', image)
    b1 = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=3 / 9, sigmaY=6 / 9)
    b2 = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=6 / 9, sigmaY=3 / 9)
    image1 = b1 - b2
    image2 = b2 - b1
    cv2.imshow('image1', image1)
    cv2.waitKey(0)

    cv2.imshow('image2', image2)
    cv2.waitKey(0)


    b3 = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=7 / 9, sigmaY=14 / 9)
    b4 = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=14 / 9, sigmaY=7 / 9)
    image3 = b3 - b4
    image4 = b4 - b3
    cv2.imshow('image3', image3) 
    cv2.waitKey(0)
    cv2.imshow('image4', image4)
    cv2.waitKey(0)

    b5 = cv2.GaussianBlur(image, ksize=(13, 13), sigmaX=13 / 9, sigmaY=26 / 9)
    b6 = cv2.GaussianBlur(image, ksize=(13, 13), sigmaX=26 / 9, sigmaY=13 / 9)
    image5 = b6 - b5
    image6 = b5 - b6
    cv2.imshow('image5', image5)
    cv2.waitKey(0)
    cv2.imshow('image6', image6)

    print("max values", np.max(image), np.max(image1), np.max(image2), np.max(image3), np.max(image4), np.max(image5), np.max(image6))

    image = torch.stack([torch.as_tensor(image1), torch.as_tensor(image2),
                         torch.as_tensor(image3), torch.as_tensor(image4),
                         torch.as_tensor(image5), torch.as_tensor(image6)])
    # ignore values below 50
    image = F.threshold(image, threshold=50.0/255.0, value=0)
    cv2.imshow("img",  F.threshold(torch.as_tensor(image1), threshold=255.0/50.0, value=0).numpy())
    cv2.waitKey(0)
    print(imgage)
    # temporal_image = torch.zeros([self.timesteps] + list(image.shape))
    # intensity to latency
    # for t in range(self.timesteps):
    #     temporal_image[t] = self.temporal_transform(image)
    # # return temporal_image.sign().byte()

    # print("out", out.shape, out)
    break


def main():
    # load MNIST dataset, filter with DoG filters and perform intensity to latency encoding
    transform = transforms.Compose([transforms.ToTensor(), SpikeTransform(), ])

    train_dataset = datasets.MNIST(root='data', train=True, transform=transform,
                                   download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform,
                                  download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                              shuffle=False)

    net = nn.Sequential(
            # use convolution layer from spikingjelly because it is already
            # wrapped to support both step modes and works better when saving models

            # S1
            layer.Conv2d(6, 30, kernel_size=5, bias=False),
            OneSpikeIF(v_threshold=15, surrogate_function=surrogate.ATan(), ),
            # C1  - pooling (first to spike / maximum potential)
            # spike-based
            layer.MaxPool2d(kernel_size=2, stride=2),

            # S2
            layer.Conv2d(30, 250, kernel_size=3, bias=False),
            OneSpikeIF(v_threshold=10, surrogate_function=surrogate.ATan()),
            # C2
            # spike-based
            layer.MaxPool2d(kernel_size=3, stride=3),

            # S3
            layer.Conv2d(250, 200, kernel_size=5, bias=False),
            OneSpikeIF(v_threshold=float('inf'),
                       surrogate_function=surrogate.ATan()),
            # C3 - global pooling, neurons are preassigned to a digit
            # potential-based
            layer.MaxPool2d(kernel_size=5, stride=0),
            # [TODO] max pooling probably doesn't work here out of the box
            # (because neurons don't spike and we want to pool the internal potential)
    )

    # use multi-step mode for faster training
    functional.set_step_mode(net, step_mode='m')

    # using the cupy back end can speed up training
    if use_cupy:
        functional.set_backend(net, backend='cupy')

    instances_stdp = (layer.Conv2d,)

    stdp_learners = []

    for i in range(net.__len__()):
        if isinstance(net[i], instances_stdp):
            stdp_learners.append(
                    learning.STDPLearner(step_mode="m", synapse=net[i],
                                         sn=net[i + 1], tau_pre=1.0,
                                         tau_post=1.0))

    params_stdp = []
    for m in net.modules():
        if isinstance(m, instances_stdp):
            for p in m.parameters():
                params_stdp.append(p)

    # train in a layer by layer fashion
    optimizer = SGD(params_stdp, lr=0.004, momentum=0.)

    for epoch in range(0, 1):
        # train
        for frame, label in train_loader:
            optimizer.zero_grad()
            frame = frame.to('cpu:0')
            frame = frame.transpose(0,
                                    1)  # [N, T, C, H, W] -> [T, N, C, H, W]  # ...

            optimizer.zero_grad()

            print("frame", frame.shape)

            y = net(frame)
            print("y", y)
            loss = F.cross_entropy(y, label)
            print("loss", loss)
            loss.backward()

            optimizer.zero_grad()

            for i in range(stdp_learners.__len__()):
                stdp_learners[i].step(on_grad=True)

            optimizer.step()

            functional.reset_net(net)
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].reset()


        # test
        with torch.no_grad():
            for frame, label in test_loader:
                frame = frame.to('cpu:0')
                frame = frame.transpose(0, 1)

# main()