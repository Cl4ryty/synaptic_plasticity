import torch.nn as nn
from neuron import OneSpikeIF
from spikingjelly.activation_based import layer, surrogate




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
