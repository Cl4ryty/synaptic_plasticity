import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import cv2
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import encoding, functional

from network import Network
from plasticity import get_k_winners, STDP
import numpy as np
import os
import datetime


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
        image = torchvision.transforms.functional.convert_image_dtype(image, dtype=torch.int8)

        # ignore values below 50
        image = F.threshold(image, threshold=50, value=0)
        temporal_image = torch.zeros([self.timesteps]+list(image.shape))

        # intensity to latency
        for t in range(self.timesteps):
            temporal_image[t] = self.temporal_transform(image)
        return temporal_image


use_cupy = True
BATCH_SIZE = 2
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
    transform = transforms.Compose([transforms.ToTensor(), SpikeTransform(), ])

    train_dataset = datasets.MNIST(root='data', train=True, transform=transform,
                                   download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform,
                                  download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                              shuffle=False)


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

    training = [[[0, s1_training_epochs], 1], [[0, s2_training_epochs], 2],
                [[0, s3_training_epochs], 3]]


    training = training[training_layer-1:]
    print("training", training, training_layer)
    if start_epoch != 0:
        training[0][0][
            0] = start_epoch
        print("training changed start", training, training_layer)

    # Initialize TensorBoard writer
    writer = SummaryWriter(
        'runs/experiment_1')  # [TODO] make this unique for each run? Or keep the same for continuing training at the same step

    for [start_epoch, end_epoch], training_layer in training:
        if training_layer == 3:
            running_correct = 0
            running_incorrect = 0
            running_no_spikes = 0
            batch_count = 0

        for epoch in range(start_epoch, end_epoch+1):
            # train
            perf = torch.tensor([0,0,0]) # correct, wrong, silence
            for batch, (frame, label) in enumerate(train_loader):
                print(f" batch {batch}")
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

                    if batch!=0 and batch % scale_learning_rate_after_batches == 0:
                        ap = torch.tensor(net.stdp2.learning_rate[0][0].item(), device=net.stdp2.learning_rate[0][0].device) * 2
                        ap = torch.min(ap, torch.tensor([0.15]))
                        an = ap * -0.75
                        net.stdp2.update_all_learning_rate(ap.item(), an.item())
    
                if training_layer == 3:
                    # update weights according to STDP
                    # s3_learner.step(on_grad=False, eligible_neurons=winners)
    
                    # update weights according to RSTDP
                    for reward in rewards:
                        if reward == 1:
                            # reward
                            net.stdp3(pad2_out, n3_out, potential3, kwta=1)
                        else:
                            # punish
                            net.anti_stdp3(pad2_out, n3_out, potential3, kwta=1)

                    # get hits and misses for this batch
                    batch_hits = torch.sum(rewards == 1)
                    batch_misses = torch.sum(rewards == -1)

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


                # reset the network and the learners after each batch
                functional.reset_net(net)
                # net.stdp1.reset()
                # net.stdp2.reset()
                # net.stdp3.reset()
                # net.anti_stdp3.reset()

            print(f"epoch: {epoch}, training_layer: {training_layer}")
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

                    c1_out, n1_out, p1_out, pad1_out, potential1, c2_out, n2_out, p2_out, pad2_out, potential2, c3_out, n3_out, potential3 = net(frame)

                    decisions = net.get_decision()

                    number_correct = torch.sum(decisions == label)
                    number_no_spike = torch.sum(decisions == -1)
                    number_incorrect = decisions.__len__() - number_correct - number_no_spike
                    running_correct += number_correct
                    running_incorrect += number_incorrect
                    running_no_spikes += number_no_spike
                    batch_count += 1

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