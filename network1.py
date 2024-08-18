import torch
import torch.nn as nn
from neuron import OneSpikeIF
from spikingjelly.activation_based import layer, surrogate




class Network(nn.Module):
    def __init__(self, number_of_classes=None):
        super(Network, self).__init__()
        self.number_of_classes = number_of_classes
        self.class_labels = None
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
            # potential-based → this is done using the get decision function
        )
        # initialize conv weights
        def custom_weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.8, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.net[0].apply(custom_weights_init)
        self.net[4].apply(custom_weights_init)
        self.net[5].apply(custom_weights_init)

    def __len__(self):
        return len(self.net)


    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)

            # keep a record of the spikes of each layer for training purposes
            if i == 1:
                self.s1_spikes = x
            elif i == 5:
                self.s2_spikes = x
            elif i == 9:
                self.s3_spikes = x

        return x

    def get_decision(self, number_of_classes=None):
        if number_of_classes is None and self.number_of_classes is None:
            raise ValueError("Need to specify number of classes")

        if number_of_classes is not None:
            self.number_of_classes = number_of_classes

        if self.class_labels is None or self.class_labels.shape != self.net[-1].v[0].shape:
            to_repeat = torch.flatten(
                    self.net[-1].v[0]).__len__() // number_of_classes
            self.class_labels = torch.reshape(
                torch.repeat(torch.arange(number_of_classes), to_repeat),
                self.net[-1].v[0].shape)

        # get the number of output neurons
        # assign neurons to classes
        number_of_batches = self.net[-1].v.shape[0]


        decisions = torch.ones(number_of_batches) * -1 # return -1 if the network didn't spike and the maximum potential is 0

        # [TODO] see if this can be done more efficiently without a for loop through clever indexing
        # do this for each batch
        for batch in range(number_of_batches):
            max_potential, max_index = torch.max(self.net[-1].v[batch])
            if max_potential > 0.0:
                decisions[batch] = self.class_labels[max_index]

        return decisions


