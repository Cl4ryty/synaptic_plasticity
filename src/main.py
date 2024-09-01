import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional
from network import Network
import utils

# General settings and training parameters
use_cupy = True
batch_size = 100
s1_training_iterations = 100000
s2_training_iterations = 200000
s3_training_iterations = 40000000
valuation_after_iterations = 60000
tensorboard_dir = 'runs/experiment_1'  # Important: Increment the number for a new experiment
checkpoint_dir = 'checkpoints/experiment_1'
run_neuromorphic = False


# parsing command line arguments, previous settings are defaults and used if no arguments are passed
parser = argparse.ArgumentParser(description='Script to running training loop to train the SNN to classify digits')
parser.add_argument('--use_cupy', type=bool, default=use_cupy, help='Flag to use CuPy (default: True)')
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size for training (default: 100)')
parser.add_argument('--s1_training_iterations', type=int, default=s1_training_iterations, help='Training iterations for layer S1 (default: 100000)')
parser.add_argument('--s2_training_iterations', type=int, default=s2_training_iterations, help='Training iterations for layer S2 (default: 200000)')
parser.add_argument('--s3_training_iterations', type=int, default=s3_training_iterations, help='Training iterations for layer S3 (default: 40000000)')
parser.add_argument('--valuation_after_iterations', type=int, default=valuation_after_iterations, help='Iterations after which to perform valuation (default: 60000)')
parser.add_argument('--tensorboard_directory', type=str, default=tensorboard_dir, help='Tensorboard directory for logging (default: runs/experiment_1)')
parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir, help='Directory for saving checkpoints (default: checkpoints/experiment_1)')
parser.add_argument('--run_neuromorphic', type=bool, default=run_neuromorphic, help='Flag to train on N-MNIST instead of MNIST (default: False)')

args = parser.parse_args()
use_cupy = args.use_cupy
batch_size = args.batch_size
s1_training_iterations = args.s1_training_iterations
s2_training_iterations = args.s2_training_iterations
s3_training_iterations = args.s3_training_iterations
valuation_after_iterations = args.valuation_after_iterations
tensorboard_dir = args.tensorboard_directory
checkpoint_dir = args.checkpoint_dir
run_neuromorphic = args.run_neuromorphic


def main():
    """
    This runs the main part of the experiment, which is training the reimplemented
    network using STDP and R-STDP to perform digit classification on the MNIST dataset,
    testing the performance of the network on the test set and logging the
    results for visualization and analysis.
    """



    # Load the datasets and initialize the neural network
    if run_neuromorphic:
        train_loader, test_loader = utils.load_NMNIST(batch_size=batch_size)
        net = Network(input_channels=2, number_of_classes=10)
    else:
        train_loader, test_loader = utils.load_MNIST(batch_size=batch_size)
        net = Network(input_channels=6, number_of_classes=10)

    # Use multistep mode for faster training
    functional.set_step_mode(net, step_mode='m')

    # Using the cupy back end can speed up training
    if use_cupy:
        functional.set_backend(net, backend='cupy')

    # Variables to monitor the number of correct and incorrect classifications 
    batch_hits = 0
    batch_misses = 0
    # Training dataset length
    num_train_examples = len(train_loader.dataset)

    # Frequency of scaling the learning rate
    scale_learning_rate_after_batches = 500 // batch_size

    # Calculate the number of training epochs
    s1_training_epochs = s1_training_iterations // num_train_examples
    s2_training_epochs = s2_training_iterations // num_train_examples
    s3_training_epochs = s3_training_iterations // num_train_examples
    evaluation_after_epochs = valuation_after_iterations // num_train_examples

    # Initial adaptive learning rates
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

    # Optimizer for updating the weights
    optimizer = torch.optim.SGD(net.parameters(), lr=1.0, momentum=0.)

    # Checks if there are files to load the weights from
    latest_checkpoint_path = utils.get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint_path:
        # Load the checkpoint if found
        start_epoch, training_layer = utils.load_checkpoint(net,
                                                      latest_checkpoint_path)
    else:
        # Start from scratch if no checkpoint is found
        start_epoch = 0
        training_layer = 1

    training = [[[0, s1_training_epochs, s1_training_iterations], 1],
                [[0, s2_training_epochs, s2_training_iterations], 2],
                [[0, s3_training_epochs, s2_training_iterations], 3]]


    training = training[training_layer-1:]
    if start_epoch != 0:
        training[0][0][
            0] = start_epoch
        print(f"resuming training of layer: {training_layer}, complete training configuration: {training}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)


    for [start_epoch, end_epoch, samples_to_train], training_layer in training:
        sample_counter = 0

        # If training the third layer, initialize running counters for correct, incorrect, and no-spike decisions.
        if training_layer == 3:
            running_correct = 0
            running_incorrect = 0
            running_no_spikes = 0
            batch_count = 0

        with torch.no_grad():  # Disable gradient computation (speeds up training and consumes less memory)
            for epoch in range(start_epoch, end_epoch+1):
                print(f"starting epoch: {epoch}, training_layer: {training_layer}")
                
                # Train

                # Train for only the specified number of samples (plus what is needed to fill a batch)
                # - this is more accurate than going by just epochs
                # Check this here to break out of the epoch loop and start training the next layer
                # if sample_counter >= samples_to_train:
                #     break

                for batch, (frame, label) in enumerate(train_loader):
                    print(f" batch {batch}")
                    frame = frame.float()
                    frame = frame.to('cpu:0')
                    # Swap batch and timestep dimensions, timesteps should be first
                    frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]

                    # Forward pass through the network layers
                    c1_out, n1_out, p1_out, pad1_out, potential1, c2_out, n2_out, p2_out, pad2_out, potential2, c3_out, n3_out, potential3 = net(frame)

                    # Layers are trained separately, so weights are updated accordingly
                    # for a layer only if that layer is currently being trained

                    if training_layer == 1:
                        # Update weights for layer 1 according to STDP
                        net.stdp1(frame, n1_out, potential1, kwta=5, inhibition_radius=3)
                        # Scale the learning rate after each 500 iterations
                        if batch!=0 and batch % scale_learning_rate_after_batches == 0:
                            ap = torch.tensor(net.stdp1.learning_rate[0][0].item(), device=net.stdp1.learning_rate[0][0].device) * 2
                            ap = torch.min(ap, torch.tensor([0.15]))
                            an = ap * -0.75
                            net.stdp1.update_all_learning_rate(ap.item(), an.item())
        
                    if training_layer == 2:
                        # Update weights for layer 2 according to STDP
                        net.stdp2(pad1_out, n2_out, potential2, kwta=8, inhibition_radius=2)
                        # Scale the learning rate after each 500 iterations
                        if batch!=0 and batch % scale_learning_rate_after_batches == 0:
                            ap = torch.tensor(net.stdp2.learning_rate[0][0].item(), device=net.stdp2.learning_rate[0][0].device) * 2
                            ap = torch.min(ap, torch.tensor([0.15]))
                            an = ap * -0.75
                            net.stdp2.update_all_learning_rate(ap.item(), an.item())
        
                    if training_layer == 3:
                        # Get the network decision and calculate rewards
                        decisions = net.get_decision()
                        rewards = torch.ones_like(decisions)  # Rewards are 1 if decision and label match
                        rewards[decisions!=label] = -1  # Otherwise -1

                        # Update weights according to STDP and anti-STDP
                        # STDP potentiates weights when the pre-synaptic neuron fires before the post-synaptic neuron
                        # anti-STDP potentiates weights when the post-synaptic neuron fires before the pre-synaptic neuron
                        for i, reward in enumerate(rewards):                
                            if reward == 1:
                                # Reward
                                net.stdp3(pad2_out, n3_out, potential3, kwta=1, reward_batch=i)
                            else:
                                # Punish
                                net.anti_stdp3(pad2_out, n3_out, potential3, kwta=1, reward_batch=i)

                        # Get hits and misses for this batch
                        batch_hits = torch.sum(rewards == 1)
                        batch_misses = torch.sum(rewards == -1)
                        print(f"hits: {batch_hits}, misses: {batch_misses}")

                        # Update adaptive learning rates
                        apr_adapt = apr * (batch_misses/batch_size * adaptive_int + adaptive_min)
                        anr_adapt = anr * (batch_misses/batch_size * adaptive_int + adaptive_min)
                        app_adapt = app * (batch_hits/batch_size * adaptive_int + adaptive_min)
                        anp_adapt = anp * (batch_hits/batch_size * adaptive_int + adaptive_min)
                        net.stdp3.update_all_learning_rate(apr_adapt, anr_adapt)
                        net.anti_stdp3.update_all_learning_rate(anp_adapt, app_adapt)

                        # Get the number of correct decisions, no spikes/decisions,
                        # and incorrect decisions and add them to the running values
                        number_correct = torch.sum(decisions == label)
                        number_no_spike = torch.sum(decisions == -1)
                        number_incorrect = decisions.__len__() - number_correct - number_no_spike
                        running_correct += number_correct
                        running_incorrect += number_incorrect
                        running_no_spikes += number_no_spike
                        batch_count += 1

                    # Update the model parameters
                    optimizer.step()

                    # Clamp the weights of the convolutional layers within specified ranges
                    net.conv1.weight.data.clamp_(0, 1)
                    net.conv2.weight.data.clamp_(0, 1)
                    net.conv3.weight.data.clamp_(0.2, 0.8)

                    # Reset the stateful variables after each batch
                    functional.reset_net(net)
                    net.neuron1.spiked = None
                    net.neuron2.spiked = None
                    net.neuron3.spiked = None
                    net.conv1.weight.grad = None
                    net.conv2.weight.grad = None
                    net.conv3.weight.grad = None

                    # Train for only the specified number of samples (plus what is needed to fill a batch)
                    # - this is more accurate than going by just epochs
                    sample_counter += batch_size
                    # if sample_counter >= samples_to_train:
                    #     break

                # Save a checkpoint of the model at the end of each epoch
                utils.save_checkpoint(net, epoch, training_layer, directory=checkpoint_dir)

                # Save the training accuracies
                if training_layer == 3:
                    # Calculate averages of correct, no spike, and incorrect decisions
                    correct = running_correct / batch_count
                    incorrect = running_incorrect / batch_count
                    no_spikes = running_no_spikes / batch_count
                    total = correct + incorrect + no_spikes
                    print("Training epoch", epoch, "correct", correct / total,
                        "incorrect", incorrect / total, "no_spikes",
                        no_spikes / total)

                    print(f"Training epoch: {epoch}, correct: {correct / total}, incorrect: {incorrect / total}, no spikes {no_spikes / total}")

                    # Log the results to TensorBoard
                    writer.add_scalar('Training correct percentage',
                                    correct / total, epoch)
                    writer.add_scalar('Training incorrect percentage',
                                    incorrect / total, epoch)
                    writer.add_scalar('Training no spikes percentage',
                                    no_spikes / total, epoch)


                # Test
                # Only test when training layer three as training is iterative and
                # testing therefore does not make sense when training the first two
                # layers if the other ones have not been trained yet
                if training_layer == 3 and epoch % evaluation_after_epochs == 0:
                    running_correct = 0
                    running_incorrect = 0
                    running_no_spikes = 0
                    batch_count = 0

                    for frame, label in test_loader:
                        frame = frame.to('cpu:0')
                        frame = frame.transpose(0,
                                                1)  # [N, T, C, H, W] -> [T, N, C, H, W]  # ...
                        frame = frame.float()
                        
                        # Forward pass through the network layers for validation
                        c1_out, n1_out, p1_out, pad1_out, potential1, c2_out, n2_out, p2_out, pad2_out, potential2, c3_out, n3_out, potential3 = net(frame)
                        decisions = net.get_decision()

                        # Calculate and accumulate the performance metrics for the validation batch
                        number_correct = torch.sum(decisions == label)
                        number_no_spike = torch.sum(decisions == -1)
                        number_incorrect = decisions.__len__() - number_correct - number_no_spike
                        running_correct += number_correct
                        running_incorrect += number_incorrect
                        running_no_spikes += number_no_spike
                        batch_count += 1

                        # Reset stateful variables 
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


if __name__ == '__main__':
    main()