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
> Many classes and functions in `src/utils.py` have been adapted from the [SpykeTorch framework](https://github.com/miladmozafari/SpykeTorch/blob/master/SpykeTorch/utils.py) to maintain consistency in input data encoding. <br>
> **Modifications**: If any adaptations or modifications have been made, these are documented in the respective docstrings of the classes or functions. Each docstring indicates whether the code was copied or modified and provides details about any changes made.

- ```src/utils.py``` contains functions for encoding the MNIST images into spike trains, and functions to load and save model checkpoints. 
