import matplotlib.gridspec as gridspec
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import itertools


# Initialize a color palette
# palette = sns.color_palette("pastel", 10)
palette = [
    (0.9803921568627451, 0.6901960784313725, 0.8941176470588236),
    (0.6313725490196078, 0.788235294117647, 0.9568627450980393),
    (0.5529411764705883, 0.8980392156862745, 0.6313725490196078),
    (1.0, 0.7058823529411765, 0.5098039215686274),
    (1.0, 0.6235294117647059, 0.6078431372549019),
    (0.8156862745098039, 0.7333333333333333, 1.0),
    (0.8705882352941177, 0.7333333333333333, 0.6078431372549019),
    (0.8117647058823529, 0.8117647058823529, 0.8117647058823529),
    (1.0, 0.996078431372549, 0.6392156862745098),
    (0.7254901960784313, 0.9490196078431372, 0.9411764705882353),
]
markers = ['s', 'o', '^', 'd', 'X', 'v', '*', 'p', '<', '>']
markers_palette = [
    (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
    (0.00784313725490196, 0.24313725490196078, 1.0),
    (0.082, 0.631, 0.176),
    (1.0, 0.48627450980392156, 0.0),
    (0.9098039215686274, 0.0, 0.043137254901960784),
    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
    (0.6235294117647059, 0.2823529411764706, 0.0),
    (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
    (1.0, 0.7686274509803922, 0.0),
    (0.0, 0.8431372549019608, 1.0),
]


WINDOW = 20


def plot_comparison_from_files_with_padding(file_paths, metric_index, labels, title, save_path,
                                            fig_size=(4, 4), x_max=None, x_mul=20, y_min=0, y_max=None,
                                            locs=dict(loc='upper right'),
                                            highlight=False):

    y_max = 1.0 if y_max == None else y_max

    fig = plt.figure(figsize=fig_size)

    sns.set_theme(style="ticks")

    custom_legend_handles = []

    # Find the maximum length among all datasets to ensure uniform plotting
    max_length = 0
    avg_datasets = []
    all_datasets = []
    for f, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            avg_data = data[0]
            all_data = data[1]
            avg_datasets.append(avg_data[metric_index])
            all_datasets.append(all_data[metric_index])
            max_length = max(max_length, len(avg_data[metric_index]))

    # Print std
    for label, avg_dataset, all_dataset in zip(labels, avg_datasets, all_datasets):
        min_run_length = min(len(run) for run in all_dataset)
        trimmed_all_dataset = [run[:min_run_length] for run in all_dataset]
        # shape: (runs, min_run_length)
        all_arr = np.array(trimmed_all_dataset)
        # print(all_arr.shape)

        # per-epoch std
        # shape: (epochs,)
        # TODO
        std_per_epoch = np.nanstd(all_arr[:, 180:200], axis=0)
        max_std = np.nanmax(std_per_epoch)
        min_std = np.nanmin(std_per_epoch)
        mean_std = np.nanmean(std_per_epoch)
        median_std = np.nanmedian(std_per_epoch)
        # print(f"[{label}] all_data std per epoch: {std_per_epoch.tolist()}")
        print(f"[{label:10.10}] std stats — max: {max_std:.4f}, min: {min_std:.4f}, mean: {mean_std:.4f}, median: {median_std:.4f}")

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
                            y=data_set, alpha=0.0625, s=20, color=color, legend=False)

        # Using pandas to handle NaNs gracefully in lineplot
        padded_avg_data = padded_avg_data if x_max == None else padded_avg_data[:x_max]

        # window-based avg
        if WINDOW != 0:
            padded_avg_data = (
                pd.Series(padded_avg_data)
                .rolling(window=WINDOW, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )

        df = pd.DataFrame(
            {'Epoch': x_axis, 'Value': padded_avg_data, 'Group': label})
        sns.lineplot(x='Epoch', y='Value', data=df, style='Group', zorder=2,
                     dashes=False,
                     linewidth=2.0 if i == 0 and highlight else 1.0,
                     alpha=1.0 if i == 0 and highlight else 0.6, color=marker_color)
        #  markers=marker, markersize=4, markeredgewidth=0.5)

        # Overlay scatterplot at a reduced frequency for markers
        # Sampling for marker density
        sampled_df = df.iloc[[(40+(i)*20) % x_max]]
        sns.scatterplot(x='Epoch', y='Value', data=sampled_df, zorder=3,
                        marker=marker, color=marker_color, s=50, edgecolor='black',
                        legend=False)

        # Create a custom legend handle for this dataset
        line = mlines.Line2D([], [], color=marker_color, marker=marker,
                             linestyle='-',
                             #  linewidth=0.75,
                             linewidth=1.5 if i == 0 and highlight else 0.75,
                             markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                             label=label)
        custom_legend_handles.append(line)

    # Setting the y-axis limits
    plt.ylim(y_min, y_max)

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
    if locs is not None:
        # plt.legend(**locs)
        plt.legend(handles=custom_legend_handles, **locs)
    else:
        ax = plt.gca()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.tight_layout()
    plt.grid(linewidth=0.25)

    plot_path = os.path.join(save_path, f"{title}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}")


def expand_data_numpy(data, repeat):
    expanded_data = [np.repeat(sublist, repeat).tolist() for sublist in data]
    return expanded_data


if __name__ == '__main__':
    plot_directory = './save/avg_objects'

    metric_index = 1  # 0 for "loss", 1 for "acc"

    """
    0. Extending
    """
    multiplier = 1
    with open(f'{plot_directory}/nn_cifar_cnn_.pkl', 'rb') as file:
        data = pickle.load(file)
        extended_avg_sgd = expand_data_numpy(data[0], multiplier)
        extended_all_sgd = [expand_data_numpy(
            d, multiplier) for d in data[1]]

    with open(f'{plot_directory}/nn_cifar_cnn__extended.pkl', 'wb') as f:
        pickle.dump([extended_avg_sgd, extended_all_sgd], f)

    # TODO
    # for iid in [1, 0]:
    for iid in [0]:
        save_path = './save/combined/iid' if iid == 1 else './save/combined/non_iid'

        """
        1. Performance
        """
        print("\nConvergence")
        title = 'Convergence'
        file_paths = [
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2.pkl',
            f'{plot_directory}/fedasync_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_S16_A0.6.pkl',
            f'{plot_directory}/fedavg_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0.pkl',
            f'{plot_directory}/nn_cifar_cnn__extended.pkl'
        ]
        labels = [
            'FRAIN',
            'BRAIN',
            'FedAsync',
            'FedAvg',
            'SGD'
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=200,
            y_min=0.5,
            # y_max=0.725,
            # locs=dict(loc='lower center', ncol=2),
            locs=dict(loc='lower right', ncol=1),
            highlight=True
        )

        """
        2-1. Byzantine
        - FRAIN :    Nullifiers=10
        - BRAIN :    Nullifiers=10
        - FedAsync : Nullifiers=10
        - FedAvg :   Nullifiers=10
        """
        print("\nByzantine_Nullifiers_10")
        title = f'Byzantine_Nullifiers_10'
        file_paths = [
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10_SZ0_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10_SZ0_D0.55_W4_S16_TH0.2.pkl',
            f'{plot_directory}/fedasync_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10_S16_A0.6.pkl',
            f'{plot_directory}/fedavg_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z10.pkl'
        ]
        labels = [
            'FRAIN',
            'BRAIN',
            'FedAsync',
            'FedAvg'
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=200,
            y_min=0.0,
            # y_max=0.725,
            # locs=dict(loc='lower center', ncol=2),
            locs=dict(
                loc='lower right',
                bbox_to_anchor=(1.0, 0.2),
                ncol=2
            ),
            highlight=True
        )

        """
        2-2. Score Byzantine
        - 0, 5, 10, 11, 15

        Distruptors (with 5 Nullifiers)
        """
        print("\nByzantine_Distruptors_at_Nullifiers_5")
        title = f'Byzantine_Distruptors_at_Nullifiers_5'
        file_paths = [
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ5_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ5_D0.55_W4_S16_TH0.2.pkl',
        ]
        labels = [
            'FRAIN',
            'BRAIN'
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=200,
            y_min=0.0,
            # y_max=0.725,
            # locs=dict(loc='lower center', ncol=2),
            locs=dict(
                loc='lower right',
                bbox_to_anchor=(1.0, 0.2),
                ncol=1
            ),
            highlight=True
        )

        # """
        # 3-1. LERP vs SLERP (@ Stale & Drift)
        # - FRAIN (SLERP, hinge)
        # - FRAIN (SLERP, constant)
        # - FRAIN (LERP,  hinge)
        # - BRAIN (LERP,  constant)
        # """
        # print("\nL_vs_SL_all")
        # title = f'L_vs_SL_all'
        # file_paths = [
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_hinge_a10.0_b4.0_c16.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_constant_a0.0_b0.0_c4.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_lerp_hinge_a10.0_b4.0_c16.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_lerp_constant_a0.0_b0.0_c4.0.pkl',
        # ]
        # labels = [
        #     # 'SLERP',
        #     # 'LERP',
        #     'SLERP (Hinge)',
        #     'SLERP (Const)',
        #     'LERP  (Hinge)',
        #     'LERP  (Const)',
        # ]
        # print(file_paths)
        # plot_comparison_from_files_with_padding(
        #     file_paths, metric_index, labels, title, save_path,
        #     fig_size=(4, 3.0),
        #     # fig_size=(4, 3.5),
        #     x_max=200,
        #     y_min=0.275,
        #     y_max=0.925,
        #     # locs=dict(loc='lower center', ncol=2),
        #     locs=dict(
        #         loc='lower right',
        #         # bbox_to_anchor=(1.0, 0.2),
        #         ncol=1
        #     ),
        #     highlight=False
        # )

        """
        3. LERP vs SLERP (@ Stale & Drift)
        - FRAIN (SLERP, hinge)
        - FRAIN (LERP,  hinge)
        """
        print("\nL_vs_SL")
        title = f'L_vs_SL'
        file_paths = [
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_lerp_hinge_a10.0_b4.0_c16.0.pkl',
        ]
        labels = [
            'SLERP',
            'LERP'
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=200,
            y_min=0.675,
            y_max=0.925,
            # locs=dict(loc='lower center', ncol=2),
            locs=dict(
                loc='lower right',
                # bbox_to_anchor=(1.0, 0.2),
                ncol=1
            ),
            highlight=True
        )

        # """
        # 4-1. Staleness Panelty Functions
        # - constant
        # - poly
        # - hinge
        # """
        # print("\nPenalty_Functions")
        # title = f'Penalty_Functions'
        # file_paths = [
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_hinge_a10.0_b2.0_c16.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_hinge_a10.0_b4.0_c16.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_hinge_a10.0_b8.0_c16.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_poly_a0.5_b0.0_c4.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_constant_a0.0_b0.0_c4.0.pkl',
        # ]
        # labels = [
        #     'Hinge_2',
        #     'Hinge_4',
        #     'Hinge_8',
        #     'Polynomial',
        #     'Constant',
        # ]
        # print(file_paths)
        # plot_comparison_from_files_with_padding(
        #     file_paths, metric_index, labels, title, save_path,
        #     fig_size=(4, 3.0),
        #     # fig_size=(4, 3.5),
        #     x_max=200,
        #     y_min=0.25,
        #     y_max=0.95,
        #     # locs=dict(loc='lower center', ncol=2),
        #     locs=dict(
        #         loc='lower right',
        #         # bbox_to_anchor=(1.0, 0.2),
        #         ncol=1
        #     ),
        #     highlight=False
        # )

        """
        4. Staleness Panelty Functions
        - constant
        - poly
        - hinge
        """
        print("\nPenalty")
        title = f'Penalty'
        file_paths = [
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_poly_a0.5_b0.0_c4.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_constant_a0.0_b0.0_c4.0.pkl',
        ]
        labels = [
            'Hinge',
            'Polynomial',
            'Constant',
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=200,
            y_min=0.675,
            y_max=0.925,
            # locs=dict(loc='lower center', ncol=2),
            locs=dict(
                loc='lower right',
                # bbox_to_anchor=(1.0, 0.2),
                ncol=1
            ),
            highlight=True
        )

        # """
        # 5-1. FastSync
        # - FRAIN
        # """
        # print("\nDrift_FRAIN")
        # title = f'Drift_FRAIN'
        # file_paths = [
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR5_slerp_hinge_a10.0_b4.0_c16.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_hinge_a10.0_b4.0_c16.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR15_slerp_hinge_a10.0_b4.0_c16.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR21_slerp_hinge_a10.0_b4.0_c16.0.pkl',
        # ]
        # labels = [
        #     '0',
        #     '5',
        #     '11',
        #     '15',
        #     '21'
        # ]
        # print(file_paths)
        # plot_comparison_from_files_with_padding(
        #     file_paths, metric_index, labels, title, save_path,
        #     fig_size=(4, 3.0),
        #     # fig_size=(4, 3.5),
        #     x_max=200,
        #     y_min=0.5,
        #     # locs=dict(loc='lower center', ncol=2),
        #     locs=dict(
        #         loc='lower right',
        #         # bbox_to_anchor=(1.0, 0.2),
        #         ncol=1
        #     ),
        #     highlight=False
        # )

        # """
        # 5-2. FastSync
        # - BRAIN
        # """
        # print("\nDrift_BRAIN")
        # title = f'Drift_BRAIN'
        # file_paths = [
        #     # f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR0_lerp_constant_a0.0_b0.0_c4.0.pkl',
        #     f'{plot_directory}/brain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2.pkl',

        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR5_lerp_constant_a0.0_b0.0_c4.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_lerp_constant_a0.0_b0.0_c4.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR15_lerp_constant_a0.0_b0.0_c4.0.pkl',
        #     f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR21_lerp_constant_a0.0_b0.0_c4.0.pkl',
        # ]
        # labels = [
        #     '0',
        #     '5',
        #     '11',
        #     '15',
        #     '21'
        # ]
        # print(file_paths)
        # plot_comparison_from_files_with_padding(
        #     file_paths, metric_index, labels, title, save_path,
        #     fig_size=(4, 3.0),
        #     # fig_size=(4, 3.5),
        #     x_max=200,
        #     y_min=0.5,
        #     # locs=dict(loc='lower center', ncol=2),
        #     locs=dict(
        #         loc='lower right',
        #         # bbox_to_anchor=(1.0, 0.2),
        #         ncol=1
        #     ),
        #     highlight=False
        # )

        """
        5. FastSync
        - FRAIN (DR=11)
        - BRAIN (DR=11)
        """
        print("\nDrift")
        title = f'Drift'
        file_paths = [
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR11_lerp_constant_a0.0_b0.0_c4.0.pkl',
        ]
        labels = [
            'FRAIN',
            'BRAIN',
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=200,
            y_min=0.575,
            y_max=0.925,
            # locs=dict(loc='lower center', ncol=2),
            locs=dict(
                loc='lower right',
                # bbox_to_anchor=(1.0, 0.2),
                ncol=1
            ),
            highlight=True
        )

        """
        5-3. FastSync
        - FRAIN (DR 0 to 21)
        - Distruptor 5, Nullifier 5
        """
        print("\nDrift_Byzantines")
        title = f'Drift_Byzantines'
        file_paths = [
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ5_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ5_D0.55_W4_S16_TH0.2_DR5_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ5_D0.55_W4_S16_TH0.2_DR11_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ5_D0.55_W4_S16_TH0.2_DR15_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z5_SZ5_D0.55_W4_S16_TH0.2_DR21_slerp_hinge_a10.0_b4.0_c16.0.pkl',
        ]
        labels = [
            '0',
            '5',
            '11',
            '15',
            '21',
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=200,
            y_min=0.575,
            y_max=0.925,
            # locs=dict(loc='lower center', ncol=2),
            locs=dict(
                loc='lower right',
                # bbox_to_anchor=(1.0, 0.2),
                ncol=2
            ),
            highlight=True
        )

        """
        6. 101
        - FRAIN
        """
        print("\nLarge_Scale")
        title = f'Large_Scale'
        file_paths = [
            # f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B4096_Z0_SZ0_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B2048_Z0_SZ0_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B1024_Z0_SZ0_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B512_Z0_SZ0_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
            f'{plot_directory}/frain_cifar_cnn_C0.1_iid{iid}_E9.9_B256_Z0_SZ0_D0.55_W4_S16_TH0.2_DR0_slerp_hinge_a10.0_b4.0_c16.0.pkl',
        ]
        labels = [
            # '5',
            '11',
            '21',
            '51',
            '101',
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=200,
            y_min=0.2,
            # y_max=0.725,
            # locs=dict(loc='lower center', ncol=2),
            locs=dict(loc='lower right', ncol=1),
            highlight=False
        )
