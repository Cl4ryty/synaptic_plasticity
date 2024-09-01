# Replication study: Bio-inspired digit recognition using reward-modulated spike-timing-dependent plasticity in deep convolutional networks

**Group 7:** Krishnedu Bose, Alexander Ditz, Hannah Köster, Tim Kapferer, Hanna Willkomm

This is our project for the **Modelling of Synaptic Plasticity** course at the University of Osnabrück. We developed an implementation of the model introduced in the paper ["Bio-inspired digit recognition using reward-modulated spike-timing-dependent plasticity in deep convolutional networks" by Mozafari et al.](https://www.sciencedirect.com/science/article/abs/pii/S0031320319301906) The original model was implemented using the [SpykeTorch](https://arxiv.org/pdf/1903.02440) framework. In our project, we reimplemented the model using the [Spikingjelly](https://arxiv.org/pdf/2310.16620) framework by Fang et al.

## Setup and Installation

**1. Open a terminal and clone this repository:** <br>
```
git clone https://github.com/Cl4ryty/synaptic_plasticity.git 
cd synaptic_plasticity
```

**2. Create a virtual environment (optional but recommended):** <br>
We use conda to create the virtual environment. To install conda you can follow the official [documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 
```
conda create -n synaptic_plasticity python=3.9 nb_conda_kernels
conda activate synaptic_plasticity 
```

**3. Install the required dependencies:**
```
pip install -r requirements.txt
```

## Running the Project
### Training the Model

To train the network, execute the `main.py` file. 

When starting a new experiment, remember to increment the experiment number in line 14 to ensure that TensorBoard generates a separate plot for each experiment. For example:

```python
tensorboard_directory = 'runs/experiment_4'  # Important: Increment the number for a new experiment
```
<br>

Ensure you select the correct checkpoints for your experiments. We provide trained models for MNIST (experiment_1) and N-MNIST (experiment_2). 
If you want to train the model from scratch, create a new folder (e.g., `experiment_3`) in the `checkpoints` directory and update the `checkpoint_dir` in line 15 accordingly. For example:

```python
checkpoint_dir = 'checkpoints/experiment_3'
```
To use the N-MNIST dataset, set `run_neuromorphic` in line 16 to `True`. Set it to `False` if you wish to use the original MNIST dataset.


### Using and Playing Around with the Model

In the notebook `mnist_model_demo.ipynb`, we go through each step to run the model with visualizations to make the process more engaging and informative.
Run it locally by starting `jupyter lab`, opening the notebook and running its contents there, or <a href="https://colab.research.google.com/github/Cl4ryty/synaptic_plasticity/blob/separated_data_loading/mnist_model_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Viewing the logged results with TensorBoard
During training of our different model configurations some metrics were logged using [TensorBoard](https://www.tensorflow.org/tensorboard/get_started). The log files are located in the ```runs``` directory and can be opened with TensorBoard by starting it with the path of the log files to display provided as `--logdir` argument. For example, to display the logged metrics for the model trained on MNIST run the following command 
```
tensorboard --logdir runs/experiment_1
```  

## Project Structure and File Descriptions

```
synaptic_plasticity
├── .gitignore
├── README.md
├── requirements.txt
├── mnist_model_demo.ipynb
├── LICENSE
│
├── checkpoints
│   ├── experiment_1
│   └── experiment_2
│
├── runs
│   ├── experiment_1
│   ├── experiment_2
│   └── experiment_3
│
└── src
    ├── main.py
    ├── network.py
    ├── neuron.py
    ├── plasticity.py
    ├── plotting.py
    └── utils.py

```

- ```mnist_model_demo.ipynb``` provides a demonstration of how to use the model
- ```checkpoints``` contains the checkpoints saved during training for each of different training runs of our models
    - ```checkpoints/experiment_1``` contains the checkpoints for the model trained on MNIST
    - ```checkpoints/experiment_2``` contains the checkpoints for the model trained on N-MNIST
- ```runs``` contains the tensorboard log files for the different experiment runs
    - ```runs/experiment_1``` contains the log files for the run of the reimplement model trained on MNIST
    - ```runs/experiment_2``` contains the log files for the run of the reimplement model trained on N-MNIST
    - ```runs/experiment_3``` contains the log files for the run of the SpykeTorch model - this is used as a benchmark for comparison
- ```src/main.py``` contains the training loop to train the spiking neural network.
- ```src/network.py``` specifies the network architecture.
- ```src/neuron.py``` specifies the dynamics of the Integrate-and-Fire (IF) neuron model.
- ```src/plasticity.py``` contains the implementation of the learning algorithm - the STDP as described by [Mozafari et al.](https://www.sciencedirect.com/science/article/abs/pii/S0031320319301906) - as well the function to select the k winners eligible for plasticity.
- ```src/plotting.py``` contains functions for plotting the logged accuracies.
> [!IMPORTANT]
> Many classes and functions in `src/utils.py` have been adapted from the [SpykeTorch framework](https://github.com/miladmozafari/SpykeTorch/blob/master/SpykeTorch/utils.py) to maintain consistency in input data encoding. <br>
> **Modifications**: If any adaptations or modifications have been made, these are documented in the respective docstrings of the classes or functions. Each docstring indicates whether the code was copied or modified and provides details about any changes made.

- ```src/utils.py``` contains functions for encoding the MNIST images into spike trains, and functions to load and save model checkpointLogs

[TODO] add folder for report, including figures/plots
