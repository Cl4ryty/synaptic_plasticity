import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import encoding, functional
from network import Network
from plasticity import get_k_winners, STDP
import numpy as np
import os
import datetime
import utils


# class SpikeTransform:
#     def __init__(self, timesteps=15):
#         self.temporal_transform = encoding.LatencyEncoder(timesteps)
#         self.count = 0
#         self.timesteps = timesteps

#     def __call__(self, image):
#         image = torch.as_tensor(image).squeeze().numpy()
#         b1 = cv2.GaussianBlur(image,ksize=(3,3),sigmaX=3/9,sigmaY=6/9)
#         b2 = cv2.GaussianBlur(image,ksize=(3,3),sigmaX=6/9,sigmaY=3/9)
#         image1 = b1 - b2
#         image2 = b2 - b1

#         b3 = cv2.GaussianBlur(image,ksize=(7,7),sigmaX=7/9,sigmaY=14/9)
#         b4 = cv2.GaussianBlur(image,ksize=(7,7),sigmaX=14/9,sigmaY=7/9)
#         image3 = b3 - b4
#         image4 = b4 - b3

#         b5 = cv2.GaussianBlur(image,ksize=(13,13),sigmaX=13/9,sigmaY=26/9)
#         b6 = cv2.GaussianBlur(image,ksize=(13,13),sigmaX=26/9,sigmaY=13/9)
#         image5 = b6 - b5
#         image6 = b5 - b6

#         image = torch.stack([torch.as_tensor(image1), torch.as_tensor(image2), torch.as_tensor(image3), torch.as_tensor(image4), torch.as_tensor(image5), torch.as_tensor(image6)])
#         image = torchvision.transforms.functional.convert_image_dtype(image, dtype=torch.int8)

#         # ignore values below 50
#         image = F.threshold(image, threshold=50, value=0)
#         temporal_image = torch.zeros([self.timesteps]+list(image.shape))

#         # intensity to latency
#         for t in range(self.timesteps):
#             temporal_image[t] = self.temporal_transform(image)
#         return temporal_image


# Following code copied from SpykeTorch implementation
def local_normalization(input, normalization_radius, eps=1e-12):
    r"""Applies local normalization. on each region (of size radius*2 + 1) the mean value is computed and the
    intensities will be divided by the mean value. The input is a 4D tensor.

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
    means = F.conv2d(y,kernel,padding=normalization_radius) + eps # computes means
    y = y/means # normalization
    # swap minibatch with channels
    y.squeeze_(1)
    y.unsqueeze_(0)
    return y

class Intensity2Latency:
    r"""Applies intensity to latency transform. Spike waves are generated in the form of
    spike bins with almost equal number of spikes.

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
# Copy from SpykeTorch end


use_cupy = True
BATCH_SIZE = 100
s1_training_iterations = 100000
s2_training_iterations = 200000
s3_training_iterations = 40000000
valuation_after_iterations = 60000


def save_checkpoint(model, epoch, training_layer, directory='checkpoints'):
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

def main():
    # load MNIST dataset, filter with DoG filters and perform intensity to latency encoding
    # transform = transforms.Compose([transforms.ToTensor(), SpikeTransform(), ])

    # train_dataset = datasets.MNIST(root='data', train=True, transform=transform,
    #                                download=True)
    # test_dataset = datasets.MNIST(root='data', train=False, transform=transform,
    #                               download=True)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                                            shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
    #                                           shuffle=False)
    kernels = [ utils.DoGKernel(3,3/9,6/9),
            utils.DoGKernel(3,6/9,3/9),
            utils.DoGKernel(7,7/9,14/9),
            utils.DoGKernel(7,14/9,7/9),
            utils.DoGKernel(13,13/9,26/9),
            utils.DoGKernel(13,26/9,13/9)]
    # threshold changed to 30 instead of 50; otherwise only spikes for the first 3 time steps
    filter = utils.Filter(kernels, padding = 6, thresholds = 30)
    s1c1 = S1C1Transform(filter)   

    data_root = "data"
    train_dataset = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1))
    test_dataset = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    net = Network(number_of_classes=10)


    # use multistep mode for faster training
    functional.set_step_mode(net, step_mode='m')

    # using the cupy back end can speed up training
    if use_cupy:
        functional.set_backend(net, backend='cupy')

    batch_hits = 0
    batch_misses = 0
    batch_size = BATCH_SIZE

    num_train_examples = len(train_dataset)

    scale_learning_rate_after_batches = 500 // BATCH_SIZE

    s1_training_epochs = s1_training_iterations // num_train_examples
    s2_training_epochs = s2_training_iterations // num_train_examples
    s3_training_epochs = s3_training_iterations // num_train_examples
    valuation_after_epochs = valuation_after_iterations // num_train_examples

    # initial adaptive learning rates
    apr = net.stdp3.learning_rate[0][0].item()
    anr = net.stdp3.learning_rate[0][1].item()
    app = net.anti_stdp3.learning_rate[0][1].item()
    anp = net.anti_stdp3.learning_rate[0][0].item()

    adaptive_min = 0
    adaptive_int = 1
    apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
    anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
    app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
    anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

    optimizer = torch.optim.SGD(net.parameters(), lr=1.0, momentum=0.)

    # check if there are files to load the weights from
    checkpoint_dir = 'checkpoints'
    latest_checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint_path:
        # Load the checkpoint if found
        start_epoch, training_layer = load_checkpoint(net,
                                                      latest_checkpoint_path)

    else:
        # Start from scratch if no checkpoint is found
        start_epoch = 0
        training_layer = 1

    training = [[[0, s1_training_epochs, s1_training_iterations], 1],
                [[0, s2_training_epochs, s2_training_iterations], 2],
                [[0, s3_training_epochs, s2_training_iterations], 3]]


    training = training[training_layer-1:]
    print("training", training, training_layer)
    if start_epoch != 0:
        training[0][0][
            0] = start_epoch
        print("training changed start", training, training_layer)

    # Initialize TensorBoard writer
    writer = SummaryWriter(
        'runs/experiment_2')  # [TODO] make this unique for each run? Or keep the same for continuing training at the same step

    for [start_epoch, end_epoch, samples_to_train], training_layer in training:
        sample_counter = 0
        if training_layer == 3:
            running_correct = 0
            running_incorrect = 0
            running_no_spikes = 0
            batch_count = 0

        with torch.no_grad():
            for epoch in range(start_epoch, end_epoch+1):
                print(f"starting epoch: {epoch}, training_layer: {training_layer}")
                # train
                perf = torch.tensor([0,0,0]) # correct, wrong, silence

                # train for only the specified number of samples (plus what is needed to fill a batch)
                # - this is more accurate than going by just epochs
                # check this here to break out of the epoch loop and start straining the next layer
                sample_counter += batch_size
                # if sample_counter >= samples_to_train:
                #     break

                for batch, (frame, label) in enumerate(train_loader):
                    print(f" batch {batch}")
                    frame = frame.float()
                    frame = frame.to('cpu:0')
                    frame = frame.transpose(0,
                                            1)  # [N, T, C, H, W] -> [T, N, C, H, W]  # ...
        
                    c1_out, n1_out, p1_out, pad1_out, potential1, c2_out, n2_out, p2_out, pad2_out, potential2, c3_out, n3_out, potential3 = net(frame)

                    if training_layer == 3:
                        decisions = net.get_decision()
                        rewards = torch.ones_like(decisions)  # rewards are 1 if decision and label match
                        rewards[decisions!=label] = -1  # -1 otherwise

                    if training_layer == 1:
                        # update weights according to STDP
                        # spikes_pre, spikes_post, potentials, winners=None, kwta=1, inhibition_radius=0
                        net.stdp1(frame, n1_out, potential1, kwta=5, inhibition_radius=3)
                        if batch!=0 and batch % scale_learning_rate_after_batches == 0:
                            ap = torch.tensor(net.stdp1.learning_rate[0][0].item(), device=net.stdp1.learning_rate[0][0].device) * 2
                            ap = torch.min(ap, torch.tensor([0.15]))
                            an = ap * -0.75
                            net.stdp1.update_all_learning_rate(ap.item(), an.item())
        
                    if training_layer == 2:
                        # update weights according to STDP
                        net.stdp2(pad1_out, n2_out, potential2, kwta=8, inhibition_radius=2)
                        # print(net.neuron2.spiked[0])

                        if batch!=0 and batch % scale_learning_rate_after_batches == 0:
                            ap = torch.tensor(net.stdp2.learning_rate[0][0].item(), device=net.stdp2.learning_rate[0][0].device) * 2
                            ap = torch.min(ap, torch.tensor([0.15]))
                            an = ap * -0.75
                            net.stdp2.update_all_learning_rate(ap.item(), an.item())
        
                    if training_layer == 3:
                        # update weights according to STDP
                        # s3_learner.step(on_grad=False, eligible_neurons=winners)

                        # print(potential3.shape, n3_out.shape)
                        # print(n3_out[:, 0, 0])
                        
        
                        # update weights according to RSTDP
                        # print("rewards ", rewards.shape)
                        # print(pad2_out.shape, n3_out.shape, potential3.shape)
                        for i, reward in enumerate(rewards):                
                            if reward == 1:
                                # reward
                                
                                net.stdp3(pad2_out, n3_out, potential3, kwta=1, reward_batch=i)
                            else:
                                # punish
                                net.anti_stdp3(pad2_out, n3_out, potential3, kwta=1, reward_batch=i)

                        # get hits and misses for this batch
                        batch_hits = torch.sum(rewards == 1)
                        batch_misses = torch.sum(rewards == -1)
                        print("hits ", batch_hits, "miss ", batch_misses)

                        #update adaptive learning rates
                        apr_adapt = apr * (batch_misses/BATCH_SIZE * adaptive_int + adaptive_min)
                        anr_adapt = anr * (batch_misses/BATCH_SIZE * adaptive_int + adaptive_min)
                        app_adapt = app * (batch_hits/BATCH_SIZE * adaptive_int + adaptive_min)
                        anp_adapt = anp * (batch_hits/BATCH_SIZE * adaptive_int + adaptive_min)
                        net.stdp3.update_all_learning_rate(apr_adapt, anr_adapt)
                        net.anti_stdp3.update_all_learning_rate(anp_adapt, app_adapt)

                        number_correct = torch.sum(decisions == label)
                        number_no_spike = torch.sum(decisions == -1)
                        number_incorrect = decisions.__len__() - number_correct - number_no_spike
                        running_correct += number_correct
                        running_incorrect += number_incorrect
                        running_no_spikes += number_no_spike
                        batch_count += 1

                    optimizer.step()
                    # TODO: lower_bound and upper_bound in STDP not actually used yet
                    net.conv1.weight.data.clamp_(0, 1)
                    net.conv2.weight.data.clamp_(0, 1)
                    net.conv3.weight.data.clamp_(0.2, 0.8)
                    # reset the network and the learners after each batch
                    # print(net.conv1.weight[0,0])
                    functional.reset_net(net)
                    # net.stdp1.reset()
                    # net.stdp2.reset()
                    # net.stdp3.reset()
                    # net.anti_stdp3.reset()
                    net.neuron1.spiked = None
                    net.neuron2.spiked = None
                    net.neuron3.spiked = None
                    net.conv1.weight.grad = None
                    net.conv2.weight.grad = None
                    net.conv3.weight.grad = None

                    # train for only the specified number of samples (plus what is needed to fill a batch)
                    # - this is more accurate than going by just epochs
                    sample_counter += batch_size
                    # if sample_counter >= samples_to_train:
                    #     break

                save_checkpoint(net, epoch, training_layer, directory='checkpoints')

                # save training accuracies
                if training_layer == 3:
                    correct = running_correct / batch_count
                    incorrect = running_incorrect / batch_count
                    no_spikes = running_no_spikes / batch_count
                    total = correct + incorrect + no_spikes
                    print("Training epoch", epoch, "correct", correct / total,
                        "incorrect", incorrect / total, "no_spikes",
                        no_spikes / total)

                    # Log the results to TensorBoard
                    writer.add_scalar('Training correct percentage',
                                    correct / total, epoch)
                    writer.add_scalar('Training incorrect percentage',
                                    incorrect / total, epoch)
                    writer.add_scalar('Training no spikes percentage',
                                    no_spikes / total, epoch)


                # test
                if training_layer == 3 and epoch % valuation_after_epochs == 0:
                    running_correct = 0
                    running_incorrect = 0
                    running_no_spikes = 0
                    batch_count = 0

                    for frame, label in test_loader:
                        frame = frame.to('cpu:0')
                        frame = frame.transpose(0,
                                                1)  # [N, T, C, H, W] -> [T, N, C, H, W]  # ...
                        frame = frame.float()

                        c1_out, n1_out, p1_out, pad1_out, potential1, c2_out, n2_out, p2_out, pad2_out, potential2, c3_out, n3_out, potential3 = net(frame)

                        decisions = net.get_decision()

                        number_correct = torch.sum(decisions == label)
                        number_no_spike = torch.sum(decisions == -1)
                        number_incorrect = decisions.__len__() - number_correct - number_no_spike
                        running_correct += number_correct
                        running_incorrect += number_incorrect
                        running_no_spikes += number_no_spike
                        batch_count += 1

                        functional.reset_net(net)
                        net.neuron1.spiked = None
                        net.neuron2.spiked = None
                        net.neuron3.spiked = None
                        net.conv1.weight.grad = None
                        net.conv2.weight.grad = None
                        net.conv3.weight.grad = None

                    correct = running_correct / batch_count
                    incorrect = running_incorrect / batch_count
                    no_spikes = running_no_spikes / batch_count
                    total = correct+incorrect+no_spikes
                    print("Validation at epoch", epoch, "correct", correct/total, "incorrect", incorrect/total, "no_spikes", no_spikes/total)

                    # Log the results to TensorBoard
                    writer.add_scalar('Valuation correct percentage', correct/total, epoch)
                    writer.add_scalar('Valuation incorrect percentage', incorrect/total, epoch)
                    writer.add_scalar('Valuation no spikes percentage', no_spikes/total, epoch)

    # Close the tensorboard writer
    writer.close()



main()