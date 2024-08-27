# Replication study: Bio-inspired digit recognition using reward-modulated spike-timing-dependent plasticity in deep convolutional networks

**Group 7:** Krishnedu Bose, Alexander Ditz, Hannah Köster, Tim Kapferer, Hanna Willkomm

This is our project for the **Modelling of Synaptic Plasticity** course at the University of Osnabrück. We developed an implementation of the model introduced in the paper ["Bio-inspired digit recognition using reward-modulated spike-timing-dependent plasticity in deep convolutional networks" by Mozafari et al.](https://www.sciencedirect.com/science/article/abs/pii/S0031320319301906) The original model was implemented using the [SpykeTorch](https://arxiv.org/pdf/1903.02440) framework. In our project, we reimplemented the model using the [Spikingjelly](https://arxiv.org/pdf/2310.16620) framework by Fang et al.

## Setup and Installation

**1. Open a terminal and clone this repository:** <br>
```
$ git clone https://github.com/Cl4ryty/synaptic_plasticity.git 
$ cd synaptic_plasticity
```

**2. Create a virtual environment (optional but recommended):** <br>
We use conda to create the virtual environment. To install conda you can follow the official [documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 
```
$ conda create -n synaptic_plasticity python=3.9
$ conda activate synaptic_plasticity 
```

**3. Install the required dependencies:**
```
$ pip install -r requirements.txt
```

## Running the Project

TODO: Separate training and testing into two distinct functions, so we can also just test the network

## Project Structure and File Descriptions

```
synaptic_plasticity
│   README.md
│   .gitignore  
│
└───src
    │   main.py
    │   network.py
    │   neuron.py
    │   plasticity.py
    │   utils.py
```

- ```src/main.py``` contains the training loop to train the spiking neural network.
- ```src/network.py``` specifies the network architecture.
- ```src/neuron.py``` specifies the dynamics of the Integrate-and-Fire (IF) neuron model.
- ```src/plasticity.py``` contains the implementation of the learning algorithm - the STDP as described by [Mozafari et al.](https://www.sciencedirect.com/science/article/abs/pii/S0031320319301906) - as well the function to select the k winners eligible for plasticity.
> [!IMPORTANT]
> src/utils.py is mostly copied from [SpykeTorch](https://github.com/miladmozafari/SpykeTorch/blob/master/SpykeTorch/utils.py) to ensure that the input data is encoded the same way. The only difference is that we turned off the accumulation of spikes to restrict an input neuron to produce a single spike only.
- ```src/utils.py``` contains functions for encoding the MNIST images into spike trains. 
