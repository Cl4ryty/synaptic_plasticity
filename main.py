import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import cv2

from spikingjelly.activation_based import encoding, functional, layer

from network1 import Network
from plasticity import get_k_winners_batched, CustomSTDPLearner


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

def main():
    # load MNIST dataset, filter with DoG filters and perform intensity to latency encoding
    transform = transforms.Compose([transforms.ToTensor(), SpikeTransform(), ])

    train_dataset = datasets.MNIST(root='data', train=True, transform=transform,
                                   download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform,
                                  download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                              shuffle=False)


    net = Network()


    # use multistep mode for faster training
    functional.set_step_mode(net, step_mode='m')

    # using the cupy back end can speed up training
    if use_cupy:
        functional.set_backend(net, backend='cupy')


    # STDP for S1
    s1_learner = CustomSTDPLearner(step_mode="m", synapse=net.net[0], sn=net.net[1], alpha=1, beta=0,
                              ar_p=0.004, ar_n=-0.003, ap_p=0.0, ap_n=0.0,
                              phi_r=1.0, phi_p=0)

    # STDP for S2
    s2_learner = CustomSTDPLearner(step_mode="m", synapse=net.net[4], sn=net.net[5],
                              alpha=1, beta=0, ar_p=0.004, ar_n=-0.003,
                              ap_p=0.0, ap_n=0.0, phi_r=1.0, phi_p=0)

    # RSTDP for S3
    s3_learner = CustomSTDPLearner(step_mode="m", synapse=net.net[8], sn=net.net[9],
                              alpha=1, beta=0, ar_p=0.004, ar_n=-0.003,
                              ap_p=0.0005, ap_n=-0.004, phi_r=0, phi_p=0)


    # [TODO] train in a layer by layer fashion

    training_layer = 1

    for epoch in range(0, 1):
        # train
        for frame, label in train_loader:
            frame = frame.to('cpu:0')
            frame = frame.transpose(0,
                                    1)  # [N, T, C, H, W] -> [T, N, C, H, W]  # ...

            print("frame", frame.shape)
            print("label", label.shape)
            print(net.net[1])
            print(net.net[1].v)

            s = torch.ones((15,6,28,28))
            s1 = torch.zeros((15, 6, 28, 28))
            s = torch.stack((s, s1))
            print(s.shape)
            s = s.permute(1, 0, 2, 3, 4)
            print(s.shape)

            y = net(s)
            print("y", y.shape)
            print("v_Seq", net.net[1].v_seq.shape)
            print("neurons", net.net[1])
            # print("net spikes", torch.nonzero(net.spikes))
            # print("max potentials", torch.max(net.net[1].v_seq[:,0,:,:,:], dim=1))


            if training_layer == 1:
                # get k winners eligible for plasticity
                winners = get_k_winners_batched(net.s1_spikes, net.net[1].v_seq, k=5,
                                          r=3)
                print("winners shape for stdp",winners.shape)
                # update weights according to STDP
                s1_learner.step(on_grad=False, eligible_neurons=winners)

            if training_layer == 2:
                # get k winners eligible for plasticity
                winners = get_k_winners_batched(net.s2_spikes,
                                                net.net[5].v_seq, k=8, r=2)
                # update weights according to STDP
                s2_learner.step(on_grad=False, eligible_neurons=winners)

            if training_layer == 3:
                # get k winners eligible for plasticity
                winners = get_k_winners_batched(net.s3_spikes, net.net[9].v_seq, k=5,
                                          r=3)
                # update weights according to STDP
                s3_learner.step(on_grad=False, eligible_neurons=winners)

                # [TODO] update weights according to RSTDP
                # s3_learner.step_rstdp(rewards, batch_hits, batch_misses, batch_size, on_grad=False, eligible_neurons=winners)


            # reset the network and the learners after each batch
            functional.reset_net(net)
            s1_learner.reset()
            s2_learner.reset()
            s3_learner.reset()


main()