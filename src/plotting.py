import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_data(log_path, scalar_tag):
    """
    Extract scalar data from a TensorBoard log file.
    :param log_path: string specifying the path to the log file.
    :param scalar_tag: string containing the tag of the scalar to extract.
    :return:
        epochs: list of epochs
        values: list of scalar values
    """
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()

    if scalar_tag in event_acc.Tags()['scalars']:
        scalar_events = event_acc.Scalars(scalar_tag)
        epochs = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        return epochs, values
    else:
        print(f"Tag {scalar_tag} not found in logs.")
        return [], []

def plot_accuracies(model_dirs, tags, color_mapping, label_mapping, baseline, number_of_epochs, save_as=None):
    """
    Plot the baseline and the accuracies of multiple models/runs provided as
    tensorboard records in the provided directories in the specified colors,
    with the specified labels, for the specified number of epochs.

    :param model_dirs: list of strings specifying the directories containing the tensorboard checkpoints.
    :param tags: list of strings specifying the names of the tags (as they are named in the checkpoints) to plot (e.g. test and validation accuracy).
    :param color_mapping: dictionary mapping colors to each model_dir-tag combination with first level keys being the  model_dir strings, which each map to a dictionary with the tags as keys and desired colors as values.
    :param label_mapping: dictionary mapping model_dir and label strings to strings that are to be displayed in the legend.
    :param baseline: float value to plot as baseline
    :param number_of_epochs: int value specifying the number of epochs to plot, epochs will be plotted from 0 to this number.
    :param save_as (optional): string specifying the path/filename under which the plot should be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Control font sizes - this is to be able to choose appropriate font sizes for larger figures
    SIZE_DEFAULT = 14
    SIZE_LARGE = 16
    plt.rc("font", family="Sans")
    plt.rc("font", weight="normal")
    plt.rc("font", size=SIZE_DEFAULT)
    plt.rc("axes", titlesize=SIZE_LARGE)
    plt.rc("axes", labelsize=SIZE_LARGE)
    plt.rc("xtick", labelsize=SIZE_DEFAULT)
    plt.rc("ytick", labelsize=SIZE_DEFAULT)

    # Plot the baseline
    ax.plot([0, number_of_epochs], [baseline*100, baseline*100], label="Baseline",
            color="lightgray", linestyle="--", linewidth=1, )

    for model_dir in model_dirs:
        for tag in tags:
            for root, dirs, files in os.walk(model_dir):
                epoch_list = []
                value_list = []
                for file in files:
                    if "events.out.tfevents" in file:
                        log_file = os.path.join(root, file)
                        epochs, values = extract_scalar_data(log_file, tag)
                        print(epochs, values)
                        epoch_list += epochs
                        value_list += values

                epochs = np.asarray(epoch_list)
                values = np.asarray(value_list)
                # Sort by epoch - necessary because files may not be read in the correct order
                sorted_indices = np.argsort(epochs)

                ax.plot(epochs[sorted_indices], values[sorted_indices]*100, label=f"{label_mapping[model_dir]} - {label_mapping[tag]}", color=color_mapping[model_dir][tag], linewidth=2)

    # Hide the all but the bottom spines (axis lines)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_bounds(0, number_of_epochs)

    # Put legend centered next to the plot to the right side, and remove the box around it
    ax.legend(bbox_to_anchor=(1.0, 0.5), loc='center left',
                             borderaxespad=0., frameon=False)

    # Set the labels
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    plt.tight_layout()

    # Save the plot with the specified name if one was passed
    if save_as is not None:
        plt.savefig(save_as, dpi=300)

    plt.show()


def create_final_plot():
    """
    Wrapper function which called the plot_accuracies function with the correct
    paths and values to create our final plot for the report.
    """

    # Directory paths for the TensorBoard logs
    model_dirs = ['runs/experiment_1', 'runs/experiment_2', 'runs/experiment_3']

    # Tags for training and testing accuracies
    tags = ['Training correct percentage', 'Valuation correct percentage']

    # Mapping of names to colors
    color_mapping = {
            "runs/experiment_1": {'Training correct percentage': "#CE96FF",
                                  'Valuation correct percentage': "#BF55EC"},
            "runs/experiment_2": {'Training correct percentage': "#72D5B3",
                                  'Valuation correct percentage': "#03A678"},
            "runs/experiment_3": {'Training correct percentage': "#F9E9A8",
                                  'Valuation correct percentage': "#E0B765"}, }

    # Mopping of paths and tag names to label to use in plot legend
    label_mapping = {"runs/experiment_1": "1", "runs/experiment_2": "2",
            "runs/experiment_3": "3", 'Training correct percentage': "training",
            'Valuation correct percentage': "testing"}

    # baseline to plot - this is the highest accuracy reported in the paper
    baseline = 0.972

    # number of epochs to plot
    number_of_epochs = 20

    plot_accuracies(model_dirs, tags, color_mapping, label_mapping, baseline,
                    number_of_epochs, save_as='accuracies.png')

