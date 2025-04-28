import os
import pickle
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


def categorize_files(directory_path):
    categories = defaultdict(list)
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pkl') and os.path.isfile(os.path.join(directory_path, file_name)):
            options_part, _, timestamp = file_name.rpartition('_')
            method, _, ramains = options_part.partition('_')
            dataset, _, ramains = ramains.partition('_')
            network, _, ramains = ramains.partition('_')
            epoch, _, ramains = ramains.partition('_')

            options_part = method + '_' + dataset + '_' + network + '_' + ramains
            categories[options_part].append(file_name)
    return categories


def load_and_average_files(category, files, load_path, save_path):
    all_data = defaultdict(list)  # To store data by their index for averaging

    # Track the max length seen for each index (0 for test_loss_collect, 1 for test_acc_collect)
    max_lengths = defaultdict(int)

    for file_name in files:
        with open(os.path.join(load_path, file_name), 'rb') as file:
            data = pickle.load(file)
            for i, values in enumerate(data):
                all_data[i].append(values)
                max_lengths[i] = max(max_lengths[i], len(values))

    # Averaging, considering different lengths by padding shorter ones with NaN
    averaged_data = []
    for i, values_list in all_data.items():
        # Pad each list in values_list to the max length observed for this index with NaN
        padded_lists = [np.pad(values, (0, max_lengths[i] - len(values)),
                               constant_values=np.nan) for values in values_list]
        # Stack and then average, ignoring NaN values
        stacked = np.vstack(padded_lists)
        averaged_data.append(np.nanmean(stacked, axis=0))

    # Saving the objects test_loss_collect and test_acc_collect:
    # file_name = f'{save_path}/{category}_{time.time()}.pkl'
    file_name = f'{save_path}/{category}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([averaged_data, list(all_data.values())], f)

    return averaged_data, list(all_data.values())


def plot_data(category, metric_index, avg_data, all_data, directory_path):
    metric_name = "Loss" if metric_index == 0 else "Accuracy"
    x_axis = np.arange(len(avg_data[metric_index]))

    plt.figure()

    # Plot averaged data
    plt.plot(x_axis, avg_data[metric_index],
             markers=True,
             label='Average', color='black', linewidth=0.75)

    # Plot individual data points
    for data_set in all_data[metric_index]:
        plt.scatter(np.arange(len(data_set)),
                    data_set, color='black', alpha=0.1, s=20)

    plt.title(f"{metric_name} for {category}")
    plt.xlabel('Round')
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.grid(linewidth=0.25)

    plt.savefig(
        f"{directory_path}/{len(all_data[metric_index])}_{category}_{metric_name}.png",
        bbox_inches='tight', dpi=300)
    plt.close()


def plot_data_seaborn(category, metric_index, avg_data, all_data, directory_path):
    metric_name = "Loss" if metric_index == 0 else "Accuracy"
    x_axis = np.arange(len(avg_data[metric_index]))

    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(6, 4))
    # sns.set_theme(style="ticks")

    # Plot averaged data as a black line using Seaborn
    sns.lineplot(x=x_axis, y=avg_data[metric_index],
                 markers=True,
                 label='Average', color='black', linewidth=0.75, zorder=2)

    # Plot individual data points using Seaborn scatterplot for each data set
    for data_set in all_data[metric_index]:
        sns.scatterplot(x=np.arange(len(data_set)),
                        y=data_set, color='black', alpha=0.1, s=20)

    plt.title(f"{metric_name} for {category}")
    plt.xlabel('Round')
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.grid(linewidth=0.25)

    plt.savefig(
        f"{directory_path}/{len(all_data[metric_index])}_{category}_{metric_name}.png",
        bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    pkl_path = './save/objects'
    avg_pkl_path = './save/avg_objects'
    png_paths = ['./save/loss', './save/acc']

    categories = categorize_files(pkl_path)

    for category, files in categories.items():
        avg_data, all_data = load_and_average_files(
            category, files, pkl_path, avg_pkl_path)
        for metric_index in range(2):
            plot_data_seaborn(category, metric_index, avg_data,
                              all_data, png_paths[metric_index])
