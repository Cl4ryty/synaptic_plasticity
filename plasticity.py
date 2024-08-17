from typing import Union

import torch
import torch.nn as nn

from spikingjelly.activation_based import monitor, base, neuron

class CustomSTDPLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str,
        synapse: Union[nn.Conv2d, nn.Linear], sn: neuron.BaseNode, alpha, beta,
        ar_p, ar_n, ap_p, ap_n, phi_r, phi_p, use_stabilizer=True,
        perform_weight_clipping=False, upper_weight_limit=1.0,
        lower_weight_limit=0.0
    ):
        super().__init__()
        self.step_mode = step_mode
        self.synapse = synapse
        self.alpha = alpha
        self.beta = beta
        self.ar_p = ar_p
        self.ar_n = ar_n
        self.ap_p = ap_p
        self.ap_n = ap_n
        self.phi_r = phi_r
        self.phi_p = phi_p
        self.use_stabilizer = use_stabilizer
        self.perform_weight_clipping = perform_weight_clipping
        self.upper_weight_limit = upper_weight_limit
        self.lower_weight_limit = lower_weight_limit

        self.in_spike_monitor = monitor.InputMonitor(synapse)
        self.out_spike_monitor = monitor.OutputMonitor(sn)

    def reset(self):
        super(CustomSTDPLearner, self).reset()
        self.in_spike_monitor.clear_recorded_data()
        self.out_spike_monitor.clear_recorded_data()

    def disable(self):
        self.in_spike_monitor.disable()
        self.out_spike_monitor.disable()

    def enable(self):
        self.in_spike_monitor.enable()
        self.out_spike_monitor.enable()

    def get_spike_order(self, spikes_pre, spikes_post):
        # get the cumulative sum for both spike tensors on the time dimension
        pre = torch.cumsum(spikes_pre, dim=0)
        post = torch.cumsum(spikes_post, dim=0)

        # assumes at max one spike per neuron
        spike_order = torch.sum(pre - post, dim=0) # sum across time - assumes at most one spike per neuron
        # clamp to get rid of negative values from when post fired before pre
        spike_order = torch.clamp(spike_order, min=0, max=1)

        return spike_order

    def step(self, on_grad: bool = True, eligible_neurons=None):
        # eligible neurons should be a mask of the same shape as weights specifying which neurons/weights should be changed through STDP
        # if eligible_neurons is not None and eligible_neurons.shape != self.synapse.weight.shape:
        #     print("eligible shape", eligible_neurons.shape, "synapse shape", self.synapse.weight.data.shape)
        #     raise ValueError("eligible neurons should be None or of the same shape is synapse.weight (i.e. the same shape as the neuron layer), (time_steps, batch_size, feature_maps, x, y)")

        length = self.in_spike_monitor.records.__len__()
        delta_w = None
        weights = self.synapse.weight.data

        for _ in range(length): # for all layers watched by this learner
            spikes_pre = self.in_spike_monitor.records.pop(0)
            spikes_post = self.out_spike_monitor.records.pop(0)
            print("pre", spikes_pre.shape,"post", spikes_post.shape)

            # if eligible neurons is none all neurons are updated, otherwise only those specified are
            if eligible_neurons is None:
                eligible_neurons = torch.ones(spikes_post.shape[1:])

            eligible_neurons = torch.zeros(spikes_post.shape[1:]) #[TODO] remove this line after testing
            #

            # pre_before_post = self.get_spike_order(spikes_pre, spikes_post)
            pre_before_post = torch.ones(spikes_post.shape[1:])
            print("pre_before_post", pre_before_post.shape)
            delta_pre = self.alpha * self.phi_r * self.ar_p + self.beta * self.phi_p * self.ap_n
            delta_post = self.alpha * self.phi_r * self.ar_n + self.beta * self.phi_p * self.ap_p
            print("delta", delta_pre, delta_post)

            # return weight update (delta_w)
            weight_updates = torch.where(pre_before_post!=0, delta_pre, delta_post)
            # mask with eligible neurons
            weight_updates = torch.where(eligible_neurons!=0, weight_updates, 0.0)
            print("weight updates", weight_updates.shape)

            # from the eligible neurons get the feature maps for which weights have to be updated
            # if a neuron is eligible for plasticity all weights in the corresponding feature map are updated
            feature_maps = torch.sum(torch.sum(weight_updates, dim=-1), dim=-1) # sum along all dimensions except the feature map one
            print("feature maps", feature_maps.shape, feature_maps)
            print("weights shape", weights.shape)
            delta_w = weights

            # [TODO] fix weight updates - each eligible neuron should result in updating the corresponding weights, have to figure out how these map though to get the shapes right

            # multiply with stabilizer term
            if self.use_stabilizer:
                delta_w = (weights - self.lower_weight_limit)*(self.upper_weight_limit - weights)

            if self.perform_weight_clipping:
                delta_w = torch.clamp(delta_w,
                                             min=self.lower_weight_limit,
                                             max=self.upper_weight_limit)



            # mask so that only the winners are updated
            delta_w = torch.where(eligible_neurons, weight_updates, 0.0)

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w

    def step_rstdp(self, rewards, batch_hits, batch_misses, batch_size, on_grad: bool = True, eligible_neurons=None):
        # rewards should be a tensor of the same shapes the number of neurons, acceptable values are -1, 1, and 0 (reward, punishment, and neutral)
        if rewards.shape != self.synapse.weight.shape:
            raise ValueError("rewards should be of the same shape as synapse.weight, (time_steps, batch_size, feature_maps, x, y)")
        if eligible_neurons is not None and eligible_neurons.shape != self.synapse.weight.shape:
            raise ValueError("eligible neurons should be None or of the same shape is synapse.weight (i.e. the same shape as the neuron layer), (time_steps, batch_size, feature_maps, x, y)")

        # calculate the phis based on hits and misses
        self.phi_r = batch_misses/batch_size
        self.phi_p = batch_hits/batch_size

        # construct alpha and beta tensors based on reward
        self.alpha = torch.where(rewards==1.0, 1.0, 0.0)
        self.beta = torch.where(rewards==-1.0, 1.0, 0.0)

        if on_grad:
            self.step(on_grad, eligible_neurons)
        else:
            return self.step(on_grad, eligible_neurons)


def get_k_winners(spikes, potentials, k, r):
    # expects unbatched input
    number_of_selected_winners = 0

    # set the potential of neurons that spiked to the maximum after the spike so that processing can be done with only the potentials
    spike_indices = torch.nonzero(spikes)
    max_potential = torch.max(torch.max(potentials), torch.as_tensor(1.0))
    for spike in spike_indices:
        potentials[spike[0]:, spike[1], spike[2], spike[3]] = max_potential

    # sum over time
    potentials_summed = torch.sum(potentials, 0)
    eligible_neurons = torch.ones_like(potentials_summed)

    # mask for winners
    winners = torch.zeros_like(potentials_summed)

    # get k neurons that spiked first
    while number_of_selected_winners < k and torch.max(potentials_summed) > 0:
        # get the neurons with the highest potential (also corresponds to the first spike if neurons spiked)
        winner_index = (potentials_summed == torch.max(potentials_summed)).nonzero()[0]
        # print("winner index", winner_index)
        # print("winners shape", winners.shape)
        winner_feature_map, winner_x, winner_y = tuple(winner_index.tolist())

        # inhibit surrounding neurons in all feature maps - use a mask for that -> set to true everywhere in the beginning and then set to false for inhibited neurons  #
        winners[winner_feature_map, winner_x, winner_y] = 1
        number_of_selected_winners += 1
        x_min = max(winner_x-r, 0)
        x_max = min(winner_x+r+1, eligible_neurons.shape[-1])
        y_min = max(winner_y-r, 0)
        y_max = min(winner_y+r+1, eligible_neurons.shape[-1])
        eligible_neurons[:, x_min:x_max, y_min:y_max] = 0
        eligible_neurons[winner_feature_map,:,:] = 0
        # multiply potentials with eligibility mask so that inhibited neurons cannot be chosen
        potentials_summed *= eligible_neurons

    return winners


def get_k_winners_batched(spikes, potentials, k, r):
    spikes = spikes.permute(1, 0, 2, 3, 4)
    # get the potentials of the neurons / get the spike times
    potentials = potentials.permute(1, 0, 2, 3, 4)
    all_winners = None

    # iterate through batches
    for batch_index in range(spikes.shape[0]):
        winners = get_k_winners(spikes[batch_index], potentials[batch_index], k, r)
        # print("winners", winners, "batch", batch_index)
        if all_winners is None:
            all_winners = winners
        else:
            all_winners = torch.stack((all_winners, winners))

    return all_winners