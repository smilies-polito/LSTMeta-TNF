import numpy as np

def cell_states_abserr(
    cell_states_trend,
    cell_states_pred,
    cell_states_order,
    cell_states_norm_factor,
):
    cell_states_trend = (cell_states_trend * cell_states_norm_factor).astype(np.int32)
    cell_states_pred = (cell_states_pred * cell_states_norm_factor).astype(np.int32)

    abserr = np.abs(cell_states_trend - cell_states_pred)
    abserr_avg = np.mean(abserr, axis=(0,1))
    abserr_std = np.std(abserr, axis=(0,1))

    abserr_dict = {}
    subt = abserr_avg.shape[0]//3

    for idx, cs in enumerate(cell_states_order):
        abserr_dict[f"{cs}/abserr_start"] = np.mean(abserr_avg[:subt, idx])
        abserr_dict[f"std/{cs}/abserr-std_start"] = np.mean(abserr_avg[:subt, idx])-np.mean(abserr_std[:subt, idx])
        abserr_dict[f"std/{cs}/abserr+std_start"] = np.mean(abserr_avg[:subt, idx])+np.mean(abserr_std[:subt, idx])

        abserr_dict[f"{cs}/abserr_mid"] = np.mean(abserr_avg[subt:2*subt, idx])
        abserr_dict[f"std/{cs}/abserr-std_mid"] =\
            np.mean(abserr_avg[subt:2*subt, idx])-np.mean(abserr_std[subt:2*subt, idx])
        abserr_dict[f"std/{cs}/abserr+std_mid"] =\
            np.mean(abserr_avg[subt:2*subt, idx])+np.mean(abserr_std[subt:2*subt, idx])
        
        abserr_dict[f"{cs}/abserr_end"] = np.mean(abserr_avg[2*subt:, idx])
        abserr_dict[f"std/{cs}/abserr-std_end"] = np.mean(abserr_avg[2*subt:, idx])-np.mean(abserr_std[2*subt:, idx])
        abserr_dict[f"std/{cs}/abserr+std_end"] = np.mean(abserr_avg[2*subt:, idx])+np.mean(abserr_std[2*subt:, idx])

    return abserr_dict