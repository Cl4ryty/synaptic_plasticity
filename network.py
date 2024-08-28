import torch
import torch.nn as nn
from neuron import SingleSpikeIFNode
from plasticity import STDP, get_k_winners
from spikingjelly.activation_based import layer, surrogate


# Defining the SNN model
class Network(nn.Module):
    def __init__(self, number_of_classes=None):
        super(Network, self).__init__()

        # Number of classes for classification
        self.number_of_classes = number_of_classes
        self.class_labels = None # Initializing Labels

        # First convolutional layer: 6 input channels, 30 output channels, 5x5 kernel size
        self.conv1 = layer.Conv2d(in_channels=6, out_channels=30, kernel_size=5)
        self.neuron1_thr = 15. # Threshold
        self.k1 = 5 # k-Winner-take-all parameter
        self.r1 = 3 # k-Winner-take-all radius
        self.neuron1 = SingleSpikeIFNode(v_threshold=self.neuron1_thr) # Spiking neuron layer
        self.pool1 = layer.MaxPool2d(kernel_size=2, stride=2) #Max pooling layer
        self.pad1 = nn.ConstantPad2d((1,1,1,1), 0) # Padding

        # Second convolutional layer: 30 input channels, 250 output channels, 3x3 kernel size
        self.conv2 = layer.Conv2d(in_channels=30, out_channels=250, kernel_size=3)
        self.neuron2_thr = 15.
        self.k2 = 8
        self.r2 = 1
        self.neuron2 = SingleSpikeIFNode(v_threshold=self.neuron2_thr)
        self.pool2 = layer.MaxPool2d(kernel_size=3, stride=3)
        self.pad2 = nn.ConstantPad2d((2,2,2,2), 0)

        # Third convolutional layer: 250 input channels, 200 output channels, 5x5 kernel size
        self.conv3 = layer.Conv2d(in_channels=250, out_channels=200, kernel_size=5)
        self.neuron3 = SingleSpikeIFNode(v_threshold=float('inf'))

        # STDP rules for synaptic plasticity in each layer
        self.stdp1 = STDP(self.conv1, (0.004, -0.003)) 
        self.stdp2 = STDP(self.conv2, (0.004, -0.003))
        self.stdp3 = STDP(self.conv3, (0.004, -0.003), use_stabilizer=False, lower_bound=0.2, upper_bound=0.8)
        self.anti_stdp3 = STDP(self.conv3, (-0.004, 0.0005), use_stabilizer=False, lower_bound=0.2, upper_bound=0.8)
        
        # Initializing the weights for each layer
        self.conv1.apply(self.custom_weights_init)
        self.conv2.apply(self.custom_weights_init)
        self.conv3.apply(self.custom_weights_init)

    # Initialize weights function
    def custom_weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.8, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Function to get the classification decision based on neuron potentials
    def get_decision(self, number_of_classes=None):
        if number_of_classes is None and self.number_of_classes is None:
            raise ValueError("Need to specify number of classes")

        if number_of_classes is not None:
            self.number_of_classes = number_of_classes

        # Create class labels if not already created or if shape mismatch
        if self.class_labels is None or self.class_labels.shape != self.neuron3.v[0].shape:
            to_repeat = torch.flatten(
                    self.neuron3.v[0]).__len__() // self.number_of_classes
            self.class_labels = torch.arange(self.number_of_classes).repeat(to_repeat)

        # get the number of output neurons
        # assign neurons to classes
        number_of_batches = self.neuron3.v.shape[0]

        # Initilizing decision tensor
        decisions = torch.ones(number_of_batches) * -1 # return -1 if the network didn't spike and the maximum potential is 0

        # # Iterate over each batch to determine the class
        for batch in range(number_of_batches):
            max_potential, max_index = torch.max(self.neuron3.v[batch]), torch.argmax(self.neuron3.v[batch])
            if max_potential > 0.0:
                decisions[batch] = self.class_labels[max_index]

        return decisions

    # Forward pass through the network
    def forward(self, input):
        
        # First Layer
        c1_out = self.conv1(input)
        n1_out = self.neuron1(c1_out)
        p1_out = self.pool1(n1_out)
        pad1_out = self.pad1(p1_out)
        potential1 = self.neuron1.v

        # Second Layer
        c2_out = self.conv2(pad1_out)
        n2_out = self.neuron2(c2_out)
        p2_out = self.pool2(n2_out)
        pad2_out = self.pad2(p2_out)
        potential2 = self.neuron2.v

        # Third Layer
        c3_out = self.conv3(pad2_out)
        n3_out = self.neuron3(c3_out)
        potential3 = self.neuron3.v

        # Record the spiking activity at the last time step
        last_time_step = potential3.sign()
        n3_out[-1,:] = last_time_step

        return c1_out, n1_out, p1_out, pad1_out, potential1, c2_out, n2_out, p2_out, pad2_out, potential2, c3_out, n3_out, potential3