import matplotlib.gridspec as gridspec
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import itertools


def plot_comparison_from_files_with_padding(file_paths, metric_index, labels, title, save_path,
                                            fig_size=(4, 4), x_max=None, x_mul=20, y_min=0, y_max=None,
                                            locs=dict(loc='upper right'),
                                            highlight=False):
    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=fig_size)
    sns.set_theme(style="ticks")

    # Initialize a color palette
    # palette = sns.color_palette("pastel", 10)
    palette = [(0.6313725490196078, 0.788235294117647, 0.9568627450980393),
               (0.5529411764705883, 0.8980392156862745, 0.6313725490196078),
               (1.0, 0.7058823529411765, 0.5098039215686274),
               (1.0, 0.6235294117647059, 0.6078431372549019),
               (0.8156862745098039, 0.7333333333333333, 1.0),
               (0.9803921568627451, 0.6901960784313725, 0.8941176470588236),
               (0.8705882352941177, 0.7333333333333333, 0.6078431372549019),
               (0.8117647058823529, 0.8117647058823529, 0.8117647058823529),
               (1.0, 0.996078431372549, 0.6392156862745098),
               (0.7254901960784313, 0.9490196078431372, 0.9411764705882353)]
    markers = ['o', '^', 'd', 'X', 's', 'v', '*', 'p', '<', '>']
    markers_palette = [(0.00784313725490196, 0.24313725490196078, 1.0),
                       (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
                       (1.0, 0.48627450980392156, 0.0),
                       (0.9098039215686274, 0.0, 0.043137254901960784),
                       (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                       (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
                       (0.6235294117647059, 0.2823529411764706, 0.0),
                       (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
                       (1.0, 0.7686274509803922, 0.0),
                       (0.0, 0.8431372549019608, 1.0)]
    custom_legend_handles = []

    # Find the maximum length among all datasets to ensure uniform plotting
    max_length = 0
    avg_datasets = []
    all_datasets = []
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            avg_data = data[0]
            all_data = data[1]
            avg_datasets.append(avg_data[metric_index])
            all_datasets.append(all_data[metric_index])
            max_length = max(max_length, len(avg_data[metric_index]))

    # Plot each dataset, padding with NaNs where necessary
    for i, (avg_dataset, all_dataset, label) in enumerate(zip(avg_datasets, all_datasets, labels)):
        # Adjust and plot data
        padded_avg_data = np.pad(
            avg_dataset, (0, max_length - len(avg_dataset)), 'constant', constant_values=np.nan)

        x_axis = np.arange(max_length)
        x_axis = x_axis if x_max == None else x_axis[:x_max]

        # Specify color from the palette
        color = palette[i]
        marker = markers[i]
        marker_color = markers_palette[i]

        # Plot individual data points using Seaborn scatterplot for each data set
        for data_set in all_dataset:
            data_set = data_set if x_max == None else data_set[:x_max]
            sns.scatterplot(x=np.arange(len(data_set)),
                            y=data_set, alpha=0.1, s=20, color=color, legend=False)

        # Using pandas to handle NaNs gracefully in lineplot
        padded_avg_data = padded_avg_data if x_max == None else padded_avg_data[:x_max]
        df = pd.DataFrame(
            {'Epoch': x_axis, 'Value': padded_avg_data, 'Group': label})
        sns.lineplot(x='Epoch', y='Value', data=df, style='Group', zorder=2,
                     dashes=False,
                     linewidth=0.75, alpha=1.0 if i == 0 or not highlight else 0.6, color=marker_color)
        #  markers=marker, markersize=4, markeredgewidth=0.5)

        # Overlay scatterplot at a reduced frequency for markers
        # Sampling for marker density
        sampled_df = df.iloc[[(10+(i)*45) % x_max]]
        sns.scatterplot(x='Epoch', y='Value', data=sampled_df, zorder=3,
                        marker=marker, color=marker_color, s=50, edgecolor='black',
                        legend=False)

        # Create a custom legend handle for this dataset
        line = mlines.Line2D([], [], color=marker_color, marker=marker,
                             linestyle='-', linewidth=0.75,
                             markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                             label=label)
        custom_legend_handles.append(line)

    # Setting the y-axis limits
    plt.ylim(y_min, 1.0 if y_max == None else y_max)

    # Customizing axes linewidth
    ax = plt.gca()  # Get the current Axes instance
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # Set the linewidth for the axes
        spine.set_color('black')

    # Adjusting x-axis labels to display values multiplied by `x_mul`
    # For example, multiply 20 for 2 quorum * 10 local epochs
    current_ticks = ax.get_xticks()
    new_tick_labels = [f"{int(tick)*x_mul}" for tick in current_ticks]
    plt.xticks(current_ticks[1:len(current_ticks)-1],
               new_tick_labels[1:len(new_tick_labels)-1])

    # metric_name = "loss" if metric_index == 0 else "acc"
    # plt.title(f"{metric_name}")
    plt.xlabel(None)
    plt.ylabel(None)
    # plt.legend(**locs)
    plt.legend(handles=custom_legend_handles, **locs)
    plt.tight_layout()
    plt.grid(linewidth=0.25)

    plot_path = os.path.join(save_path, f"{title}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_comparison_with_broken_y_axis_and_different_sizes(file_paths, metric_index, labels, title, save_path,
                                                           fig_size=(4, 4), x_max=None, x_mul=20, y_min=0, y_max=None,
                                                           locs=dict(loc='upper right'), break_point_start=None, break_point_end=None,
                                                           top_subplot_size_ratio=2, bottom_subplot_size_ratio=1,
                                                           highlight=False):
    # Initialize the figure
    plt.figure(figsize=fig_size)
    # sns.set_theme(style="ticks")

    # Create a gridspec with two rows of different heights
    gs = gridspec.GridSpec(2, 1, height_ratios=[
                           top_subplot_size_ratio, bottom_subplot_size_ratio],
                           hspace=0.0, wspace=0)

    # Create two subplots with the specified size ratios
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    # Handling the palette
    # palette = sns.color_palette("bright", len(file_paths))
    palette = [(0.6313725490196078, 0.788235294117647, 0.9568627450980393),
               (0.5529411764705883, 0.8980392156862745, 0.6313725490196078),
               (1.0, 0.7058823529411765, 0.5098039215686274),
               (1.0, 0.6235294117647059, 0.6078431372549019),
               (0.8156862745098039, 0.7333333333333333, 1.0),
               (0.9803921568627451, 0.6901960784313725, 0.8941176470588236),
               (0.8705882352941177, 0.7333333333333333, 0.6078431372549019),
               (0.8117647058823529, 0.8117647058823529, 0.8117647058823529),
               (1.0, 0.996078431372549, 0.6392156862745098),
               (0.7254901960784313, 0.9490196078431372, 0.9411764705882353)]
    markers = ['o', '^', 'd', 'X', 's', 'v', '*', 'p', '<', '>']
    markers_palette = [(0.00784313725490196, 0.24313725490196078, 1.0),
                       (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
                       (1.0, 0.48627450980392156, 0.0),
                       (0.9098039215686274, 0.0, 0.043137254901960784),
                       (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                       (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
                       (0.6235294117647059, 0.2823529411764706, 0.0),
                       (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
                       (1.0, 0.7686274509803922, 0.0),
                       (0.0, 0.8431372549019608, 1.0)]
    custom_legend_handles = []

    # Variables to store data for plotting
    max_length = 0
    avg_datasets = []
    all_datasets = []

    # Load data
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            avg_data = data[0][metric_index]
            all_data = [ds[:x_max] for ds in data[1][metric_index]]
            max_length = max(max_length, len(avg_data), *(len(ds)
                             for ds in all_data))
            avg_datasets.append(avg_data)
            all_datasets.append(all_data)

    # Determine x-axis range
    x_axis = np.arange(max_length)
    if x_max is not None:
        x_axis = x_axis[:x_max]

    for i, (avg_dataset, all_dataset, label) in enumerate(zip(avg_datasets, all_datasets, labels)):
        # Adjust and plot data
        padded_avg_data = np.pad(avg_dataset, (0, max_length - len(avg_dataset)),
                                 'constant', constant_values=np.nan)[:x_max] if x_max else avg_dataset

        x_axis = np.arange(max_length)
        x_axis = x_axis if x_max == None else x_axis[:x_max]

        # Specify color from the palette
        color = palette[i]
        marker = markers[i]
        marker_color = markers_palette[i]

        # Plot individual data points using Seaborn scatterplot for each data set
        for data_set in all_dataset:
            sns.scatterplot(x=np.arange(len(data_set)), y=data_set,
                            alpha=0.1, s=20, color=color, legend=False, ax=ax1)
            sns.scatterplot(x=np.arange(len(data_set)), y=data_set,
                            alpha=0.1, s=20, color=color, legend=False, ax=ax2)

        # Using pandas to handle NaNs gracefully in lineplot
        padded_avg_data = padded_avg_data if x_max == None else padded_avg_data[:x_max]
        df = pd.DataFrame(
            {'Epoch': x_axis, 'Value': padded_avg_data, 'Group': label})

        sns.lineplot(x='Epoch', y='Value', data=df, style='Group', zorder=2,
                     dashes=False,
                     linewidth=0.75, alpha=1.0 if i == 0 or not highlight else 0.6, color=marker_color, ax=ax1, legend=False)
        #  markers=marker, markersize=4, markeredgewidth=0.5)
        # Overlay scatterplot at a reduced frequency for markers
        # Sampling for marker density
        sampled_df = df.iloc[[(110+(i)*45) % x_max]]
        sns.scatterplot(x='Epoch', y='Value', data=sampled_df, zorder=3, ax=ax1,
                        marker=marker, color=marker_color, s=50, edgecolor='black',
                        legend=False)

        sns.lineplot(x='Epoch', y='Value', data=df, style='Group', zorder=2,
                     dashes=False,
                     linewidth=0.75, alpha=1.0 if i == 0 or not highlight else 0.6, color=marker_color, ax=ax2)
        #  markers=marker, markersize=4, markeredgewidth=0.5)
        # Overlay scatterplot at a reduced frequency for markers
        # Sampling for marker density
        sampled_df = df.iloc[[(110+(i)*45) % x_max]]
        sns.scatterplot(x='Epoch', y='Value', data=sampled_df, zorder=3, ax=ax2,
                        marker=marker, color=marker_color, s=50, edgecolor='black',
                        legend=False)

        # Create a custom legend handle for this dataset
        line = mlines.Line2D([], [], color=marker_color, marker=marker,
                             linestyle='-', linewidth=0.75,
                             markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                             label=label)
        custom_legend_handles.append(line)

    # Apply the break in y-axis between the specified points
    ax1.set_ylim(break_point_end, y_max if y_max is not None else 1.0)
    ax2.set_ylim(y_min, break_point_start)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.tick_bottom()
    ax1.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)

    # Broken axis marks
    ax1.axhline(y=break_point_end+0.0003,
                color='lightgray', linewidth=1, zorder=1)
    d = .25  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0+0.03*bottom_subplot_size_ratio/(top_subplot_size_ratio+bottom_subplot_size_ratio), 0+0.03 *
             bottom_subplot_size_ratio/(top_subplot_size_ratio+bottom_subplot_size_ratio)], transform=ax1.transAxes, zorder=3, **kwargs)
    ax2.plot([0, 1], [1-0.03*top_subplot_size_ratio/(top_subplot_size_ratio+bottom_subplot_size_ratio), 1-0.03 *
             top_subplot_size_ratio/(top_subplot_size_ratio+bottom_subplot_size_ratio)], transform=ax2.transAxes, zorder=3, **kwargs)

    # Legend and titles
    # ax2.legend(**locs)
    ax2.legend(handles=custom_legend_handles, **locs)
    # metric_name = "loss" if metric_index == 0 else "acc"
    # ax1.set_title(f"{metric_name} : {title}")
    # ax2.set_xlabel('Epoch')
    # ax1.set_ylabel(metric_name)
    # ax2.set_ylabel(metric_name)
    plt.subplots_adjust(hspace=0.0)  # Adjust space between subplots
    ax1.set_xlabel(None)
    ax1.set_ylabel(None)
    ax2.set_xlabel(None)
    ax2.set_ylabel(None)
    plt.tight_layout()
    ax1.grid(linewidth=0.25)
    ax2.grid(linewidth=0.25)

    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)  # Set the linewidth for the axes
        spine.set_color('black')
    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)  # Set the linewidth for the axes
        spine.set_color('black')

    # Adjusting x-axis labels to display values multiplied by `x_mul`
    # For example, multiply 20 for 2 quorum * 10 local epochs
    current_ticks = ax1.get_xticks()
    new_tick_labels = [f"{int(tick)*x_mul}" for tick in current_ticks]
    plt.xticks(current_ticks[1:len(current_ticks)-1],
               new_tick_labels[1:len(new_tick_labels)-1])

    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, f"{title}_broken_y_axis.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}")


def expand_data_numpy(data, repeat):
    expanded_data = [np.repeat(sublist, repeat).tolist() for sublist in data]
    return expanded_data


if __name__ == '__main__':
    plot_directory = './save/avg_objects'
    metric_index = 1  # 0 for "loss", 1 for "acc"

    for iid in [1, 0]:
        save_path = './save/combined/iid' if iid == 1 else './save/combined/non_iid'

        """
        1. Performance
        """
        multiplier = 1
        with open(f'{plot_directory}/nn_cifar_cnn_.pkl', 'rb') as file:
            data = pickle.load(file)
            extended_avg_sgd = expand_data_numpy(data[0], multiplier)
            extended_all_sgd = [expand_data_numpy(
                d, multiplier) for d in data[1]]

        with open(f'{plot_directory}/nn_cifar_cnn__extended.pkl', 'wb') as f:
            pickle.dump([extended_avg_sgd, extended_all_sgd], f)

        title = 'Convergence'
        file_paths = [
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S4_TH0.0.pkl',
            f'{plot_directory}/fedasync_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_S4_A0.6.pkl',
            f'{plot_directory}/fedavg_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0.pkl',
            f'{plot_directory}/nn_cifar_cnn__extended.pkl'
        ]
        labels = [
            'BRAIN',
            'FedAsync',
            'FedAvg',
            'SGD'
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            # fig_size=(4, 3.5), x_max=200, y_min=0.425, y_max=0.725,
            fig_size=(4, 3.5), x_max=200,
            locs=dict(loc='lower center', ncol=2),
            highlight=True)

        """
        2. Byzantine (x5)
        - BRAIN:    0 / 5 / 10 / 11 / 15
        - FedAsync: 0 / 5 / 10 / 11 / 15
        - FedAvg:   0 / 5 / 10 / 11 / 15
        """
        for b, th in zip([0, 5, 10, 11, 15], [0.0, 0.2, 0.2, 0.2, 0.2]):
            title = f'Byzantine_{b}'
            file_paths = [
                f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{b}_SZ0_D0.55_W4_S4_TH{th}.pkl',
                f'{plot_directory}/fedasync_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{b}_S4_A0.6.pkl',
                f'{plot_directory}/fedavg_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{b}.pkl'
            ]
            labels = [
                'BRAIN',
                'FedAsync',
                'FedAvg'
            ]
            # plot_comparison_with_broken_y_axis_and_different_sizes(
            print(file_paths)
            plot_comparison_from_files_with_padding(
                file_paths, metric_index, labels, title, save_path,
                fig_size=(4, 3.5), x_max=200,
                # y_min=0.075, y_max=1.0,
                locs=dict(
                    loc='lower right',
                    bbox_to_anchor=(1.0, 0.2)),
                highlight=True)
            # break_point_start=0.225, break_point_end=0.735,
            # top_subplot_size_ratio=7, bottom_subplot_size_ratio=4)

        """
        3. Threshold
        - 0.0 / 0.2 / 0.3 / 0.4 / 0.5 / 0.6
        """
        title = 'Threshold'
        file_paths = [
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10_SZ0_D0.55_W4_S4_TH0.0.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10_SZ0_D0.55_W4_S4_TH0.2.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10_SZ0_D0.55_W4_S4_TH0.3.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10_SZ0_D0.55_W4_S4_TH0.4.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10_SZ0_D0.55_W4_S4_TH0.5.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10_SZ0_D0.55_W4_S4_TH0.6.pkl'
        ]
        labels = [
            '0.0',
            '0.2',
            '0.3',
            '0.4',
            '0.5',
            '0.6'
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            # fig_size=(4, 3.5), x_max=200, y_min=0.025, y_max=0.725,
            fig_size=(4, 3.5), x_max=200,
            # y_max=0.8,
            locs=dict(
                loc='lower right',
                ncol=2,
                bbox_to_anchor=(1.0, 0.3)))

        """
        4. Score Byzantine (x2)
        - Byzantine 0: 0 / 5 / 10 / 11 / 15
        - Byzantine 5: 0 / 5 / 10 / 11 / 15
        """
        # # 0
        # title = f'Score_Byzantine_@_Z{0}'
        # file_paths = [
        #     f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{0}_SZ0_D0.55_W4_S4_TH0.0.pkl',
        #     f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{0}_SZ5_D1.0_W4_S4_TH0.0.pkl',
        #     f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{0}_SZ10_D1.0_W4_S4_TH0.0.pkl',
        #     f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{0}_SZ11_D1.0_W4_S4_TH0.0.pkl',
        #     f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{0}_SZ15_D1.0_W4_S4_TH0.0.pkl'
        # ]
        # labels = [
        #     '0',
        #     '5',
        #     '10',
        #     '11',
        #     '15'
        # ]
        # plot_comparison_from_files_with_padding(
        #     # plot_comparison_with_broken_y_axis_and_different_sizes(
        #     file_paths, metric_index, labels, title, save_path,
        #     fig_size=(4, 3.5), x_max=200,
        #     # y_min=0.075, y_max=0.615,
        #     locs=dict(
        #         loc='lower right',
        #         bbox_to_anchor=(1.0, 0.2),
        #         ncol=2))
        # # break_point_start=0.225, break_point_end=0.485,
        # # top_subplot_size_ratio=7, bottom_subplot_size_ratio=4)

        # 5
        title = f'Score_Byzantine_@_Z{5}'
        file_paths = [
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{5}_SZ0_D0.55_W4_S4_TH0.2.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{5}_SZ5_D1.0_W4_S4_TH0.2.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{5}_SZ10_D1.0_W4_S4_TH0.2.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{5}_SZ11_D1.0_W4_S4_TH0.2.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z{5}_SZ15_D1.0_W4_S4_TH0.2.pkl'
        ]
        labels = [
            '0',
            '5',
            '10',
            '11',
            '15'
        ]
        print(file_paths)
        # plot_comparison_with_broken_y_axis_and_different_sizes(
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.5), x_max=200,
            # y_min=0.075, y_max=0.615,
            locs=dict(
                loc='lower right',
                bbox_to_anchor=(1.0, 0.4),
                ncol=2))
        # locs=dict(loc='upper right', ncol=2),
        # break_point_start=0.225, break_point_end=0.485,
        # top_subplot_size_rati7=5, bottom_subplot_size_ratio=4)

        """
        5. Staleness (X2)
        - FedAsync: 4 / 8 / 16 / 32 / 64
        - BRAIN: 4 / 8 / 16 / 32 / 64
        """
        title = f'Staleness_FedAsync'
        file_paths = [
            f'{plot_directory}/fedasync_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_S4_A0.6.pkl',
            f'{plot_directory}/fedasync_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_S8_A0.6.pkl',
            f'{plot_directory}/fedasync_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_S16_A0.6.pkl',
            f'{plot_directory}/fedasync_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_S32_A0.6.pkl',
            # f'{plot_directory}/fedasync_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_S64_A0.6.pkl'
        ]
        labels = [
            '4',
            '8',
            '16',
            '32',
            # '64'
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.5), x_max=200,
            # y_min=0.075, y_max=0.615,
            locs=dict(loc='lower right', ncol=2))

        title = f'Staleness_BRAIN'
        file_paths = [
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S4_TH0.0.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S8_TH0.0.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.0.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S32_TH0.0.pkl',
            # f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S64_TH0.0.pkl'
        ]
        labels = [
            '4',
            '8',
            '16',
            '32',
            # '64'
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.5), x_max=200,
            # y_min=0.075, y_max=0.615,
            locs=dict(loc='lower right', ncol=2))

        """
        6. Quorum
        - Byzantine: 5
        - Score Byzantine: 10
        - Diff (Quorum): 0.25 (5) / [0.50 (10)] 0.55 (11) / 0.75 (15) / [0.99 (20)] 1.0 (21)
        """
        # TH 0.2
        title = f'Quorum_TH0.2'
        file_paths = [
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ10_D0.25_W4_S4_TH0.2.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ10_D0.55_W4_S4_TH0.2.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ10_D0.75_W4_S4_TH0.2.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ10_D1.0_W4_S4_TH0.2.pkl'
        ]
        labels = [
            '5',
            '11',
            '15',
            '21'
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.5), x_max=200,
            # y_max=0.8,
            locs=dict(
                loc='lower right',
                bbox_to_anchor=(1.0, 0.4),
                ncol=2))

        # TH 0.4
        if iid == 0:
            title = f'Quorum_TH0.4'
            file_paths = [
                f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ10_D0.25_W4_S4_TH0.4.pkl',
                f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ10_D0.55_W4_S4_TH0.4.pkl',
                f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ10_D0.75_W4_S4_TH0.4.pkl',
                f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ10_D1.0_W4_S4_TH0.4.pkl'
            ]
            labels = [
                '5',
                '11',
                '15',
                '21'
            ]
            print(file_paths)
            plot_comparison_from_files_with_padding(
                file_paths, metric_index, labels, title, save_path,
                fig_size=(4, 3.5), x_max=200,
                # y_max=0.8,
                locs=dict(
                    loc='lower right',
                    bbox_to_anchor=(1.0, 0.4),
                    ncol=2))

        """
        7. Fast Bootstrapping
        """
        title = 'Drift'
        file_paths = [
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S4_TH0.0.pkl',
            f'{plot_directory}/drift_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S4_TH0.0_DR5.pkl',
            f'{plot_directory}/drift_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S4_TH0.0_DR11.pkl',
            f'{plot_directory}/drift_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S4_TH0.0_DR15.pkl',
            f'{plot_directory}/drift_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S4_TH0.0_DR21.pkl'
        ]
        labels = [
            # 'BRAIN',
            # 'Fast (5)'
            # 'Fast (11)'
            # 'Fast (15)'
            # 'Fast (21)'
            0,
            5,
            11,
            15,
            21
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            # fig_size=(4, 3.5), x_max=200, y_min=0.425, y_max=0.725,
            fig_size=(4, 3.5), x_max=200,
            locs=dict(loc='lower center', ncol=3),
            highlight=True)
