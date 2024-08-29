import torch
import torch.nn.functional as fn
from torchvision import transforms
import numpy as np
import math
import os
import datetime

class FilterKernel:
    r"""
    Base class for generating image filter kernels such as Gabor, DoG, etc. Each subclass should override :attr:`__call__` function.
   
    Attribution:
    This class is adapted from the SpykeTorch framework. The original implementation can be found in the [SpykeTorch repository](https://github.com/miladmozafari/SpykeTorch/blob/master/SpykeTorch/utils.py).
    This code is licensed under the GNU General Public License (GPL) v3.0.

    """
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self):
        pass

class DoGKernel(FilterKernel):
    r"""
    Generates DoG filter kernel.
    
    Attribution:
    This class is adapted from the SpykeTorch framework. The original implementation can be found in the [SpykeTorch repository](https://github.com/miladmozafari/SpykeTorch/blob/master/SpykeTorch/utils.py).
    This code is licensed under the GNU General Public License (GPL) v3.0.

    Args:
        window_size (int): The size of the window (square window).
        sigma1 (float): The sigma for the first Gaussian function.
        sigma2 (float): The sigma for the second Gaussian function.
    """
    def __init__(self, window_size, sigma1, sigma2):
        super(DoGKernel, self).__init__(window_size)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    # returns a 2d tensor corresponding to the requested DoG filter
    def __call__(self):
        w = self.window_size//2
        x, y = np.mgrid[-w:w+1:1, -w:w+1:1]
        a = 1.0 / (2 * math.pi)
        prod = x*x + y*y
        f1 = (1/(self.sigma1*self.sigma1)) * np.exp(-0.5 * (1/(self.sigma1*self.sigma1)) * (prod))
        f2 = (1/(self.sigma2*self.sigma2)) * np.exp(-0.5 * (1/(self.sigma2*self.sigma2)) * (prod))
        dog = a * (f1-f2)
        dog_mean = np.mean(dog)
        dog = dog - dog_mean
        dog_max = np.max(dog)
        dog = dog / dog_max
        dog_tensor = torch.from_numpy(dog)
        return dog_tensor.float()

class Filter:
    r"""
    Applies a filter transform. Each filter contains a sequence of :attr:`FilterKernel` objects.
    The result of each filter kernel will be passed through a given threshold (if not :attr:`None`).
   
    Attribution:
    This class is adapted from the SpykeTorch framework. The original implementation can be found in the [SpykeTorch repository](https://github.com/miladmozafari/SpykeTorch/blob/master/SpykeTorch/utils.py).
    This code is licensed under the GNU General Public License (GPL) v3.0.

    Args:
        filter_kernels (sequence of FilterKernels): The sequence of filter kernels.
        padding (int, optional): The size of the padding for the convolution of filter kernels. Default: 0
        thresholds (sequence of floats, optional): The threshold for each filter kernel. Default: None
        use_abs (boolean, optional): To compute the absolute value of the outputs or not. Default: False

    .. note::

        The size of the compund filter kernel tensor (stack of individual filter kernels) will be equal to the 
        greatest window size among kernels. All other smaller kernels will be zero-padded with an appropriate 
        amount.
    """
    # filter_kernels must be a list of filter kernels
    # thresholds must be a list of thresholds for each kernel
    def __init__(self, filter_kernels, padding=0, thresholds=None, use_abs=False):
        tensor_list = []
        self.max_window_size = 0
        for kernel in filter_kernels:
            if isinstance(kernel, torch.Tensor):
                tensor_list.append(kernel)
                self.max_window_size = max(self.max_window_size, kernel.size(-1))
            else:
                tensor_list.append(kernel().unsqueeze(0))
                self.max_window_size = max(self.max_window_size, kernel.window_size)
        for i in range(len(tensor_list)):
            p = (self.max_window_size - filter_kernels[i].window_size)//2
            tensor_list[i] = fn.pad(tensor_list[i], (p,p,p,p))

        self.kernels = torch.stack(tensor_list)
        self.number_of_kernels = len(filter_kernels)
        self.padding = padding
        if isinstance(thresholds, list):
            self.thresholds = thresholds.clone().detach()
            self.thresholds.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
        else:
            self.thresholds = thresholds
        self.use_abs = use_abs

    # returns a 4d tensor containing the flitered versions of the input image
    # input is a 4d tensor. dim: (minibatch=1, filter_kernels, height, width)
    def __call__(self, input):
        output = fn.conv2d(input, self.kernels, padding = self.padding).float()
        if not(self.thresholds is None):
            output = torch.where(output < self.thresholds, torch.tensor(0.0, device=output.device), output)
        if self.use_abs:
            torch.abs_(output)
        return output

class CacheDataset(torch.utils.data.Dataset):
    r"""
    A wrapper dataset to cache pre-processed data. It can cache data on RAM or a secondary memory.

    Attribution:
    This class is adapted from the SpykeTorch framework. The original implementation can be found in the [SpykeTorch repository](https://github.com/miladmozafari/SpykeTorch/blob/master/SpykeTorch/utils.py).
    This code is licensed under the GNU General Public License (GPL) v3.0. 

    .. note::

        Since converting image into spike-wave can be time consuming, we recommend to wrap your dataset into a :attr:`CacheDataset`
        object.

    Args:
        dataset (torch.utils.data.Dataset): The reference dataset object.
        cache_address (str, optional): The location of cache in the secondary memory. Use :attr:`None` to cache on RAM. Default: None
    """
    def __init__(self, dataset, cache_address=None):
        self.dataset = dataset
        self.cache_address = cache_address
        self.cache = [None] * len(self.dataset)

    def __getitem__(self, index):
        if self.cache[index] is None:
            #cache it
            sample, target = self.dataset[index]
            if self.cache_address is None:
                self.cache[index] = sample, target
            else:
                save_path = os.path.join(self.cache_address, str(index))
                torch.save(sample, save_path + ".cd")
                torch.save(target, save_path + ".cl")
                self.cache[index] = save_path
        else:
            if self.cache_address is None:
                sample, target = self.cache[index]
            else:
                sample = torch.load(self.cache[index] + ".cd")
                target = torch.load(self.cache[index] + ".cl")
        return sample, target

    def reset_cache(self):
        r"""
        Clears the cached data. It is useful when you want to change a pre-processing parameter during
        the training process.
        """
        if self.cache_address is not None:
            for add in self.cache:
                os.remove(add + ".cd")
                os.remove(add + ".cl")
        self.cache = [None] * len(self)

    def __len__(self):
        return len(self.dataset)
    
class Intensity2Latency:
    r"""
    Applies intensity to latency transform. Spike waves are generated in the form of
    spike bins with almost equal number of spikes.

    Attribution:
    This class is adapted from the SpykeTorch framework. The original implementation can be found in the [SpykeTorch repository](https://github.com/miladmozafari/SpykeTorch/blob/master/SpykeTorch/utils.py).
    This code is licensed under the GNU General Public License (GPL) v3.0.

    Modification:
    - In the original implementation spikes were accumulative, i.e. a spike in timestep i was also presented in the following timesteps (i+1, i+2, ...). 
    Here the implementations is not accumulative, i.e. a spike in timestep i will not be presented again in the following timesteps. 

    Args:
        number_of_spike_bins (int): Number of spike bins (time steps).
        to_spike (boolean, optional): To generate spike-wave tensor or not. Default: False

    .. note::

        If :attr:`to_spike` is :attr:`False`, then the result is intensities that are ordered and packed into bins.
    """
    def __init__(self, number_of_spike_bins, to_spike=False):
        self.time_steps = number_of_spike_bins
        self.to_spike = to_spike
    
    # intensities is a tensor of input intensities (1, input_channels, height, width)
    # returns a tensor of tensors containing spikes in each timestep (non-accumulative)
    def intensity_to_latency(self, intensities):
        bins_intensities = []
        nonzero_cnt = torch.nonzero(intensities).size()[0]

        # Check for empty bins
        bin_size = nonzero_cnt // self.time_steps

        # Sort
        intensities_flattened = torch.reshape(intensities, (-1,))
        intensities_flattened_sorted = torch.sort(intensities_flattened, descending=True)

        # Bin packing
        sorted_bins_value, sorted_bins_idx = torch.split(intensities_flattened_sorted[0], bin_size), torch.split(intensities_flattened_sorted[1], bin_size)

        for i in range(self.time_steps):
            # modification: for each timestep a tensor with zeros is created to not accumulate spikes
            spike_map = torch.zeros_like(intensities_flattened_sorted[0])
            spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])
            spike_map = spike_map.reshape(tuple(intensities.shape))
            bins_intensities.append(spike_map.squeeze(0).float())
    
        return torch.stack(bins_intensities)

    def __call__(self, image):
        if self.to_spike:
            return self.intensity_to_latency(image).sign()
        return self.intensity_to_latency(image)
    
class S1C1Transform:
    """
    Applies a series of transformations to an image, including tensor conversion, filtering, and temporal transformation.

    Attribution:
    This class is adapted from the SpykeTorch framework. The original implementation can be found in the [SpykeTorch repository](https://github.com/miladmozafari/SpykeTorch/blob/master/MozafariDeep.py).
    This code is licensed under the GNU General Public License (GPL) v3.0.

    Args:
        filter (callable): A function or callable object that takes a tensor image as input and returns a filtered tensor.
        timesteps (int, optional): Number of timesteps for the temporal transformation. Default is 15.
    """
    def __init__(self, filter, timesteps = 15):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = Intensity2Latency(timesteps)
        self.cnt = 0
    def __call__(self, image):
        if self.cnt % 1000 == 0:
            print(self.cnt)
        self.cnt+=1
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()
    
def local_normalization(input, normalization_radius, eps=1e-12):
    r"""
    Applies local normalization. on each region (of size radius*2 + 1) the mean value is computed and the
    intensities will be divided by the mean value. The input is a 4D tensor.

    Attribution:
    This class is adapted from the SpykeTorch framework. The original implementation can be found in the [SpykeTorch repository](https://github.com/miladmozafari/SpykeTorch/blob/master/SpykeTorch/functional.py).
    This code is licensed under the GNU General Public License (GPL) v3.0.

    Args:
        input (Tensor): The input tensor of shape (timesteps, features, height, width).
        normalization_radius (int): The radius of normalization window.

    Returns:
        Tensor: Locally normalized tensor.
    """
    # computing local mean by 2d convolution
    kernel = torch.ones(1,1,normalization_radius*2+1,normalization_radius*2+1,device=input.device).float()/((normalization_radius*2+1)**2)
    # rearrange 4D tensor so input channels will be considered as minibatches
    y = input.squeeze(0) # removes minibatch dim which was 1
    y.unsqueeze_(1)  # adds a dimension after channels so previous channels are now minibatches
    means = fn.conv2d(y,kernel,padding=normalization_radius) + eps # computes means
    y = y/means # normalization
    # swap minibatch with channels
    y.squeeze_(1)
    y.unsqueeze_(0)
    return y
    
def save_checkpoint(model, epoch, training_layer, directory='checkpoints'):
    """
    Saves the current state of the model and training details to a checkpoint file.
    
    Args:
        model (torch.nn.Module): The PyTorch model whose state dictionary will be saved.
        epoch (int): The current epoch number, which will be included in the checkpoint filename.
        training_layer (int): An integer indicating the index of the layer that is currently being trained.
        directory (str, optional): The directory where the checkpoint file will be saved. Default is 'checkpoints'.
    """
    print("epoch", epoch, "tl", training_layer)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate a timestamped filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = f'checkpoint_{timestamp}_epoch_{epoch}.pth'
    filepath = os.path.join(directory, filename)

    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
            'training_layer': training_layer, }
    torch.save(checkpoint, filepath)
    print(
        f"Checkpoint saved at epoch {epoch}, training layer {training_layer}.")

def load_checkpoint(model, filename):
    """
    Loads a model checkpoint from a specified file and restores the model's state.

    Args:
        model (torch.nn.Module): The PyTorch model to which the saved state dictionary will be loaded.
        filename (str): The path to the checkpoint file to be loaded.

    Returns:
        tuple: A tuple containing:
            - epoch (int): The epoch number to resume from (incremented by 1).
            - training_layer (int): The layer that is currently being trained.    
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch'] + 1  # start from the next epoch after the saved one
        training_layer = checkpoint['training_layer']
        print(f"Checkpoint loaded: epoch {epoch}, training layer {training_layer}.")
        return epoch, training_layer
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, 1  # Resume from epoch 0 if no checkpoint is found
    
def get_latest_checkpoint(directory):
    """
    Retrieves the path to the most recent checkpoint file in the specified directory.

    Args:
        directory (str): The path to the directory where checkpoint files are stored.

    Returns:
        str or None: The full path to the most recent checkpoint file if found, otherwise `None`.
    """
    if not os.path.exists(directory):
        return None

    # List all files in the directory
    files = os.listdir(directory)

    # Filter out files that match the checkpoint filename pattern
    checkpoint_files = [file for file in files if file.startswith('checkpoint_') and file.endswith('.pth')]

    if not checkpoint_files:
        print("No checkpoint files found.")
        return None

    # Sort the checkpoint files by the timestamp in their filenames
    checkpoint_files.sort(key=lambda x: datetime.datetime.strptime(x.split('_')[1], '%Y%m%d-%H%M%S'))

    # Return the latest checkpoint file (the last one in the sorted list)
    latest_checkpoint = checkpoint_files[-1] 
    print(f"Latest checkpoint found: {latest_checkpoint}")
    return os.path.join(directory, latest_checkpoint)