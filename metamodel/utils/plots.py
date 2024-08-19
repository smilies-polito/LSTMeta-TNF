import os
import matplotlib.pyplot as plt
import numpy as np

cell_states_colors = ['#3D70F5', '#F58F33', '#755A53']

def plot_loss(loss_list, vloss_list, title, exp_dir, filename):
    plt.figure(figsize=(9,9))
    plt.plot(loss_list, linewidth=5)
    plt.plot(vloss_list, linewidth=5)
    # plt.axhline(y=min(vloss_list), color='orange', linestyle='-')
    # plt.text(-0.1, min(vloss_list), f"{min(vloss_list):.4f}", color='orange', ha='right')
    plt.suptitle(title, fontsize=30)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Error', fontsize=26)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(['Train', 'Validation'], fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, filename))

    np.save(os.path.join(exp_dir, filename.split('.')[0]+'_train.npy'), np.array(loss_list))
    np.save(os.path.join(exp_dir, filename.split('.')[0]+'_val.npy'), np.array(vloss_list))
    plt.close()

def plot_l1loss_trend(
    cell_states_trend,
    cell_states_pred,
    cell_states_order,
    cell_states_norm_factor,
    timestep,
    radius,
    exp_dir,
    filename,
    init_cell_states_trend=None
):
    cell_states_trend = cell_states_trend * cell_states_norm_factor
    cell_states_pred = cell_states_pred * cell_states_norm_factor
    if init_cell_states_trend is not None:
        cell_states_trend = np.concatenate([init_cell_states_trend, cell_states_trend], axis=-2).cumsum(axis=-2)
        cell_states_pred = np.concatenate([init_cell_states_trend, cell_states_pred], axis=-2).cumsum(axis=-2)
    else:
        cell_states_trend = cell_states_trend.astype(np.int32)
        cell_states_pred = cell_states_pred.astype(np.int32)

    l1 = np.abs(cell_states_trend - cell_states_pred)
    l1_avg = np.mean(l1, axis=(0,1))

    plt.figure(figsize=(9,9))
    plt.suptitle(f"MAE Trend - Tumor Radius {radius}", fontsize=30)

    idx = 0
    for cs, c in zip(cell_states_order, cell_states_colors):
        plt.plot(
            range(l1_avg.shape[0]), l1_avg[..., idx],
            label=cs.title(), c=c, linewidth=5
        )
        np.save(os.path.join(exp_dir, filename.split('.')[0]+f'_{cs}.npy'), l1_avg[..., idx])
        idx += 1

    plt.legend(fontsize=26)

    plt.xticks(
        np.arange(l1_avg.shape[0]+1, step=6),
        labels=timestep*np.arange(l1_avg.shape[0]+1, step=6),
        fontsize=22
    )
    plt.yticks(fontsize=22)
    plt.xlabel("Time (min)", fontsize=26)
    plt.ylabel("Number of Cells", fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, filename))
    plt.close()


def plot_ncells_rel_trend(
    cell_states_trend,
    cell_states_pred,
    cell_states_norm_factor,
    timestep,
    exp_dir,
    filename,
    init_cell_states_trend=None
):
    cell_states_trend = cell_states_trend * cell_states_norm_factor
    cell_states_pred = cell_states_pred * cell_states_norm_factor
    if init_cell_states_trend is not None:
        cell_states_trend = np.concatenate([init_cell_states_trend, cell_states_trend], axis=-2).cumsum(axis=-2)
        cell_states_pred = np.concatenate([init_cell_states_trend, cell_states_pred], axis=-2).cumsum(axis=-2)
    else:
        cell_states_trend = cell_states_trend.astype(np.int32)
        cell_states_pred = cell_states_pred.astype(np.int32)

    ncells_trend = np.sum(cell_states_trend, axis=-1)
    ncells_pred = np.sum(cell_states_pred, axis=-1)

    relerr = np.abs(ncells_trend - ncells_pred) / ncells_trend *100
    relerr_avg = np.mean(relerr, axis=(0,1))

    plt.figure(figsize=(9,9))
    plt.suptitle(f"Relative MAE Trend", fontsize=30)

    plt.plot(range(len(relerr_avg)), relerr_avg, c='#323B4B', linewidth=5)

    plt.xticks(
        np.arange(len(relerr_avg)+1, step=6),
        labels=timestep*np.arange(0, len(relerr_avg)+1, step=6),
        fontsize=22
    )
    plt.yticks(fontsize=22)
    plt.xlabel("Time (min)", fontsize=26)
    plt.ylabel("Percentage Error", fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, filename))
    np.save(os.path.join(exp_dir, filename.split('.')[0]+'.npy'), relerr_avg)
    plt.close()

def plot_sample_experiments(
    input_params_trend,
    input_params_order,
    input_params_norm_factor,
    cell_states_trend,
    cell_states_pred,
    cell_states_order,
    cell_states_norm_factor,
    n_batches,
    n_samples,
    radius,
    timestep,
    exp_dir,
    filename,
    init_cell_states_trend=None
):
    from matplotlib.lines import Line2D
    cell_states_trend = cell_states_trend * cell_states_norm_factor
    cell_states_pred = cell_states_pred * cell_states_norm_factor
    if init_cell_states_trend is not None:
        cell_states_trend = np.concatenate([init_cell_states_trend, cell_states_trend], axis=-2).cumsum(axis=-2)
        cell_states_pred = np.concatenate([init_cell_states_trend, cell_states_pred], axis=-2).cumsum(axis=-2)
    else:
        cell_states_trend = cell_states_trend.astype(np.int32)
        cell_states_pred = cell_states_pred.astype(np.int32)

    # extract n_samples random samples from cell_states, note shape is (bs, n_examples, time, n_states)
    random_batches = np.repeat(np.random.choice(cell_states_trend.shape[0], n_batches, replace=False), n_samples)
    random_samples = np.tile(np.random.choice(cell_states_trend.shape[1], n_samples, replace=True), n_batches)

    for b, s in zip(random_batches, random_samples):
        plt.figure(figsize=(9,9))
        idx = 0
        for cs, col in zip(cell_states_order, cell_states_colors):
            plt.plot(
                range(cell_states_trend.shape[2]),
                cell_states_trend[b, s, :, idx],
                label=cs.title(), c=col, linewidth=5
            )
            plt.plot(
                range(cell_states_pred.shape[2]),
                cell_states_pred[b, s, :, idx],
                label=cs.title(), linestyle='--',
                c=col, linewidth=5
            )
            idx += 1
        inp = input_params_trend[b, s] * input_params_norm_factor
        app = ""
        for idx, i in enumerate(input_params_order):
            input_str = f"{inp[idx]:.2f}".replace('.', '_')
            app += f"{i}{input_str}-"
        plt.suptitle(f"Batch {b+1} Sample {s} - Tumor Radius {radius}", fontsize=30)
        plt.xticks(
            np.arange(cell_states_trend.shape[2]+1, step=6),
            labels=timestep*np.arange(0, cell_states_trend.shape[2]+1, step=6),
            fontsize=22
        )
        plt.yticks(fontsize=22)
        plt.xlabel("Time (min)", fontsize=26)
        plt.ylabel("Number of Cells", fontsize=26)

        # Create custom legend entries
        color_legend_handles = [Line2D([0], [0], color=col, lw=4) for col in cell_states_colors]
        color_legend_labels = [cs.title() for cs in cell_states_order]
        
        linestyle_legend_handles = [
            Line2D([0], [0], color='black', lw=4, linestyle='-'),
            Line2D([0], [0], color='black', lw=4, linestyle='--')
        ]
        linestyle_legend_labels = ['Ground Truth', 'Predicted']

        # Place the legends
        legend1 = plt.legend(color_legend_handles, color_legend_labels, fontsize=22, loc='upper right')
        legend2 = plt.legend(linestyle_legend_handles, linestyle_legend_labels, fontsize=22, loc='upper left')
        plt.gca().add_artist(legend1)

        # plt.legend(fontsize=26)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"{filename}_{app}.png"))
        plt.close()
