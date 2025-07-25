import matplotlib.gridspec as gridspec
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import LogLocator, NullLocator, NullFormatter
# from brokenaxes import brokenaxes
import seaborn as sns
import pandas as pd
# import itertools


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
    (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
    (1.0, 0.48627450980392156, 0.0),
    (0.9098039215686274, 0.0, 0.043137254901960784),
    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
    (0.6235294117647059, 0.2823529411764706, 0.0),
    (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
    (1.0, 0.7686274509803922, 0.0),
    (0.0, 0.8431372549019608, 1.0),
]


def plot_comparison_from_files_with_padding(file_paths, metric_index, labels, title, save_path,
                                            fig_size=(4, 4), x_max=None, x_mul=4, y_min=0, y_max=None,
                                            locs=dict(loc='upper right'),
                                            highlight=False, arrange=None):

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

            # adjust staleness
            if arrange is not None and f < len(file_paths) - arrange:
                avg_data = np.array(avg_data)[:, 2:].tolist()
                all_data = np.array(all_data)[:, :, 2:].tolist()

            avg_datasets.append(avg_data[metric_index])
            all_datasets.append(all_data[metric_index])
            max_length = max(max_length, len(avg_data[metric_index]))

    # Print std
    for label, avg_dataset, all_dataset in zip(labels, avg_datasets, all_datasets):
        # avg_arr = np.array(avg_dataset)  # shape: (epochs,)
        all_arr = np.array(all_dataset)  # shape: (runs, epochs)

        # per-epoch std
        std_per_epoch = np.std(all_arr, axis=0)  # shape: (epochs,)
        max_std = np.max(std_per_epoch)
        min_std = np.min(std_per_epoch)
        mean_std = np.mean(std_per_epoch)
        median_std = np.median(std_per_epoch)
        # print(f"[{label}] all_data std per epoch: {std_per_epoch.tolist()}")
        print(f"[{label:10.10}] std stats — max: {max_std:.4f}, min: {min_std:.4f}, mean: {mean_std:.4f}, median: {median_std:.4f}")

        # # std between epochs
        # overall_avg_std = np.std(avg_arr)
        # print(f"[{label}] avg_data std across epochs: {overall_avg_std:.4f}")

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
                     #  linewidth=0.75,
                     linewidth=1.5 if i == 0 and highlight else 0.75,
                     alpha=1.0 if i == 0 and highlight else 0.6, color=marker_color)
        #  markers=marker, markersize=4, markeredgewidth=0.5)

        # Overlay scatterplot at a reduced frequency for markers
        # Sampling for marker density
        # sampled_df = df.iloc[[(10+(i)*45) % x_max]]
        # sampled_df = df.iloc[[(5+(i)*5) % x_max]]
        sampled_df = df.iloc[[(5+(i)*4) % x_max]]
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

    plt.yscale('log')  # TODO
    # keep only the 10^n ticks:
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=10))
    # ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())

    plt.tight_layout()
    plt.grid(linewidth=0.25)

    plot_path = os.path.join(save_path, f"{title}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_comparison_from_files_with_padding_break(file_paths, metric_index, labels, title, save_path,
                                                  fig_size=(4, 4), x_max=None, x_mul=4, y_min=1e4, y_max=1e50,
                                                  locs=dict(loc='upper right'),
                                                  highlight=False, arrange=None,
                                                  y_break_end=None, y_break_start=None, height_ratios=(1, 2)):

    # ------------------------------
    # 1) Figure / Axes 셋업
    # ------------------------------
    use_two_axes = (y_break_start is not None and y_break_end is not None)
    if use_two_axes:
        # (ax_top, ax_bottom) 순서로 받음
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=fig_size,
            gridspec_kw={
                'height_ratios': height_ratios,
                'hspace': 0.15
                # 'hspace': 0.0
            }
        )

        axes = [ax_top, ax_bottom]

        # 위쪽 축(큰 스케일)
        ax_top.set_ylim(y_break_start, y_max)
        ax_top.set_yscale('log')
        # 아래쪽 축(작은 스케일)
        ax_bottom.set_ylim(y_min, y_break_end)
        ax_bottom.set_yscale('log')

        # 스파인 숨겨서 시각적 단절
        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        # ax_top.xaxis.tick_top()

        # ax_top.xaxis.set_visible(False)
        ax_top.tick_params(
            axis='x',       # x축
            which='both',   # major+minor
            bottom=False,   # 아래쪽 눈금선 숨기기
            top=False,      # 위쪽 눈금선 숨기기
            labelbottom=False  # 아래쪽 라벨 숨기기
        )

        ax_bottom.xaxis.tick_bottom()

        # \\
        d = 0.02
        # ————— 백슬래시(\) 절단 표시 —————
        fig.canvas.draw()   # bbox 업데이트
        bbox = ax_top.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ratio = bbox.height / bbox.width
        dy = d / ratio
        kwargs = dict(transform=ax_top.transAxes, color='gray',
                      linewidth=1.5, clip_on=False)
        # 왼쪽 “\”
        ax_top.plot((-d, +d), (0+dy, 0-dy), **kwargs)
        # 오른쪽 “\”
        ax_top.plot((1-d, 1+d), (0+dy, 0-dy), **kwargs)
        # ————— 백슬래시(\) 절단 표시 —————
        fig.canvas.draw()
        bbox = ax_bottom.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ratio = bbox.height / bbox.width
        dy = d / ratio
        kwargs = dict(transform=ax_bottom.transAxes,
                      color='gray', linewidth=1.5, clip_on=False)
        # 왼쪽 “\”
        ax_bottom.plot((-d, +d), (1+dy, 1-dy), **kwargs)
        # 오른쪽 “\”
        ax_bottom.plot((1-d, 1+d), (1+dy, 1-dy), **kwargs)

        # y_break_start 지점(위 축 아래쪽)
        ax_top.axhline(y=y_break_start, color='gray', linewidth=1.5)
        # # y_break_end 지점(아래 축 위쪽)
        ax_bottom.axhline(y=y_break_end, color='gray', linewidth=1.5)

    else:
        # 단일 축
        fig = plt.figure(figsize=fig_size)
        ax_single = plt.gca()
        ax_single.set_yscale('log')
        ax_single.set_ylim(y_min, y_max)
        axes = [ax_single]

    sns.set_theme(style="ticks")

    # ------------------------------
    # 2) 데이터 로드 / 패딩
    # ------------------------------
    max_length = 0
    avg_datasets = []
    all_datasets = []
    for f, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            avg_data = data[0]
            all_data = data[1]

            # adjust staleness
            if arrange is not None and f < len(file_paths) - arrange:
                avg_data = np.array(avg_data)[:, 2:].tolist()
                all_data = np.array(all_data)[:, :, 2:].tolist()

            avg_datasets.append(avg_data[metric_index])
            all_datasets.append(all_data[metric_index])
            max_length = max(max_length, len(avg_data[metric_index]))

    # ------------------------------
    # 3) 표준편차 로그
    # ------------------------------
    for label, avg_dataset, all_dataset in zip(labels, avg_datasets, all_datasets):
        # avg_arr = np.array(avg_dataset)  # shape: (epochs,)
        all_arr = np.array(all_dataset)  # shape: (runs, epochs)
        std_per_epoch = np.std(all_arr, axis=0)
        print(f"[{label}] std stats — max: {std_per_epoch.max():.4f}, "
              f"min: {std_per_epoch.min():.4f}, mean: {std_per_epoch.mean():.4f}, "
              f"median: {np.median(std_per_epoch):.4f}")

    # ------------------------------
    # 4) 실제 플롯
    # ------------------------------
    custom_legend_handles = []
    for i, (avg_dataset, all_dataset, label) in enumerate(zip(avg_datasets, all_datasets, labels)):
        # NaN padding
        padded_avg = np.pad(
            avg_dataset, (0, max_length - len(avg_dataset)),
            'constant', constant_values=np.nan
        )
        x_axis = np.arange(max_length)
        if x_max is not None:
            x_axis = x_axis[:x_max]

        # 색/마커
        color = palette[i]
        marker = markers[i]
        marker_color = markers_palette[i]

        # 모든 축에 대해 동일하게 그리되, y축 범위에 따라 실제 보이는 건 축마다 다를 수 있음
        for ax in axes:
            # 개별 러닝 scatter
            for run_data in all_dataset:
                d = run_data[:x_max] if x_max else run_data
                sns.scatterplot(x=np.arange(len(d)), y=d, alpha=0.1, s=20,
                                color=color, legend=False, ax=ax)

            # 평균 라인
            df = pd.DataFrame({
                'Epoch': x_axis,
                'Value': padded_avg[:len(x_axis)],
                'Group': label
            })
            sns.lineplot(
                x='Epoch', y='Value', data=df, style='Group',
                dashes=False, zorder=2,
                linewidth=1.5 if i == 0 and highlight else 0.75,
                alpha=1.0 if i == 0 and highlight else 0.6,
                color=marker_color, ax=ax
            )

            # 몇 점만 marker로 찍기
            if x_max:
                sampled_df = df.iloc[[(5 + (i * 4)) % x_max]]
            else:
                sampled_df = df.iloc[[0]]
            sns.scatterplot(
                x='Epoch', y='Value', data=sampled_df, zorder=3,
                marker=marker, color=marker_color, s=50, edgecolor='black',
                legend=False, ax=ax
            )

        # 범례용 handle
        line = mlines.Line2D([], [], color=marker_color, marker=marker,
                             linestyle='-',
                             linewidth=1.5 if i == 0 and highlight else 0.75,
                             markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                             label=label)
        custom_legend_handles.append(line)

    # ------------------------------
    # 5) 축/레이블/틱/범례 꾸미기
    # ------------------------------
    if use_two_axes:
        # 두 축 모두 (로그 스케일) 메이저 틱 설정
        ax_top.yaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0,), numticks=10))
        ax_top.yaxis.set_minor_formatter(NullFormatter())
        ax_bottom.yaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0,), numticks=10))
        ax_bottom.yaxis.set_minor_formatter(NullFormatter())

        labels = ax_top.get_yticklabels()
        for l, lbl in enumerate(labels[2:-2]):
            lbl.set_visible(False if l % 2 == 0 else True)
        # ax_top.set_yticks(ax_top.get_ylim())
        # ticks = ax_top.get_yticks()
        # ax_top.set_yticks(
        #     [tck for t, tck in enumerate(ticks[1:-1]) if t % 2 == 0])

        # x축 라벨/틱은 아래쪽 축 기준
        current_ticks = ax_bottom.get_xticks()
        new_tick_labels = [f"{int(t)*x_mul}" for t in current_ticks]
        ax_bottom.set_xticks(current_ticks[1:-1])
        ax_bottom.set_xticklabels(new_tick_labels[1:-1])

        # legend는 아래 축(ax_bottom)에만 표시
        if locs is not None:
            ax_bottom.legend(handles=custom_legend_handles, **locs)
        else:
            if ax_bottom.get_legend() is not None:
                ax_bottom.get_legend().remove()
        # 위 축에 생긴 legend 있으면 제거
        if ax_top.get_legend():
            ax_top.get_legend().remove()

        ax_top.set_xlabel(None)
        ax_bottom.set_xlabel(None)
        ax_top.set_ylabel(None)
        ax_bottom.set_ylabel(None)

        # 그리드, 스파인
        for spine in ax_top.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('black')
        for spine in ax_bottom.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('black')

        ax_top.grid(linewidth=0.25)
        ax_bottom.grid(linewidth=0.25)

        ax_top.tick_params(axis='y', labelsize=8.4)
        # ax_bottom.tick_params(axis='y', labelsize=8.4)

    else:
        ax = axes[0]
        ax.yaxis.set_major_locator(LogLocator(
            base=10, subs=(1.0,), numticks=10))
        ax.yaxis.set_minor_formatter(NullFormatter())

        current_ticks = ax.get_xticks()
        new_tick_labels = [f"{int(t)*x_mul}" for t in current_ticks]
        ax.set_xticks(current_ticks[1:-1])
        ax.set_xticklabels(new_tick_labels[1:-1])

        if locs is not None:
            ax.legend(handles=custom_legend_handles, **locs)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        ax.set_xlabel(None)
        ax.set_ylabel("Value")

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('black')

        ax.grid(linewidth=0.25)

    plt.tight_layout()

    # ------------------------------
    # 6) 저장
    # ------------------------------
    plot_path = os.path.join(save_path, f"{title}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to {plot_path}")


def expand_data_numpy(data, repeat):
    expanded_data = [np.repeat(sublist, repeat).tolist() for sublist in data]
    return expanded_data


if __name__ == '__main__':
    plot_directory = './save_llm/avg_objects'

    metric_index = 0

    """
    0. Extending
    """
    multiplier = 1
    with open(f'{plot_directory}/nn_.pkl', 'rb') as file:
        data = pickle.load(file)
        extended_avg_sgd = expand_data_numpy(data[0], multiplier)
        extended_all_sgd = [expand_data_numpy(
            d, multiplier) for d in data[1]]

    with open(f'{plot_directory}/nn__extended.pkl', 'wb') as f:
        pickle.dump([extended_avg_sgd, extended_all_sgd], f)

    # TODO
    # for iid in [1, 0]:
    for iid in [0]:
        save_path = './save_llm/combined/id' if iid == 1 else './save_llm/combined/non_id'

        """
        1. Performance
        """
        title = f"Convergence"
        file_paths = [
            f'{plot_directory}/frain_C0.1_iid{iid}_E1_B16_Z0_SZ0_D0.55_W4_S4_TH0.0_DR0_slerp_constant_a0.0_b0.0_c4.0.pkl',
            f'{plot_directory}/brain_C0.1_iid{iid}_E1_B16_Z0_SZ0_D0.55_W4_S4_TH0.0.pkl',
            f'{plot_directory}/fedasync_C0.1_iid{iid}_E1_B16_Z0_S4_A0.6.pkl',
            f'{plot_directory}/fedavg_C0.1_iid{iid}_E1_B16_Z0.pkl',
            f'{plot_directory}/nn__extended.pkl'
            # f'{plot_directory}/nn_.pkl'
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
            file_paths, metric_index, labels, title,
            save_path,
            x_mul=4,  # *16
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=25,  # 30
            y_min=0.5e3,
            y_max=1.0e5,
            locs=dict(loc='lower center', ncol=2),
            highlight=True,
            arrange=2
        )

        """
        1. Byzantine
        """
        title = f"Randomizer"
        file_paths = [
            f'{plot_directory}/frain_C0.1_iid{iid}_E1_B16_Z10_SZ0_D0.55_W4_S4_TH0.2_DR0_slerp_constant_a0.0_b0.0_c4.0.pkl',
            f'{plot_directory}/brain_C0.1_iid{iid}_E1_B16_Z10_SZ0_D0.55_W4_S4_TH0.2.pkl',
            f'{plot_directory}/fedasync_C0.1_iid{iid}_E1_B16_Z10_S4_A0.6.pkl',
            f'{plot_directory}/fedavg_C0.1_iid{iid}_E1_B16_Z10.pkl',
        ]
        labels = [
            'FRAIN',
            'BRAIN',
            'FedAsync',
            'FedAvg',
        ]
        print(file_paths)
        plot_comparison_from_files_with_padding_break(
            file_paths, metric_index, labels, title,
            save_path,
            x_mul=4,
            fig_size=(4, 3.0),
            # fig_size=(4, 3.5),
            x_max=25,  # 30
            y_min=0.5e4,
            y_max=1.0e60,
            locs=dict(loc='lower center', ncol=2),
            highlight=True,
            arrange=1,
            y_break_end=1.0e5,
            y_break_start=1.0e10,
            # height_ratios=(1, 2)
            height_ratios=(5, 11)
        )

        # """Magnifying"""
        # title = f"Randomizer_Zoom"
        # plot_comparison_from_files_with_padding(
        #     file_paths, metric_index, labels, title,
        #     save_path,
        #     x_mul=4,
        #     fig_size=(4, 1.5),
        #     # fig_size=(4.5, 3.5),
        #     x_max=25,  # 30
        #     y_min=1.0e4,
        #     y_max=1.0e5,
        #     locs=None,
        #     highlight=True,
        #     arrange=1
        # )
