import pandas as pd
import json
import gzip
import os
import shutil
from pctk import multicellds

# This script defines the funztions to convert the output data from the PhysiBoSS simulations to JSON format

def input_parameters_save(simulation, output_folder, json_file_path):
    
    output_filename = os.path.join(output_folder,f'input_parameters/input_parameters_{simulation}.json')

    shutil.copy(json_file_path, output_filename)


def cell_data(simulation, output_folder):

    sim_output_folder = '../model/output'
    output_filename = os.path.join(output_folder, f'cell_data/cell_data_{simulation}.json')
    
    # Creating a MCDS reader
    reader = multicellds.MultiCellDS(output_folder=sim_output_folder)

    # Creating an iterator to load a cell DataFrame for each stored simulation time step
    df_iterator = reader.cells_as_frames_iterator()

    dict = {}

    for (t, df_cells) in df_iterator:
        
        df_cells['current_phase'] = df_cells['current_phase'].astype(str)
        df_cells.loc[df_cells.current_phase=='14.0', 'current_phase'] = 'alive'
        df_cells.loc[df_cells.current_phase=='100.0', 'current_phase'] = 'apoptotic'
        df_cells.loc[df_cells.current_phase=='101.0', 'current_phase'] = 'necrotic'
        df_cells = df_cells[['x_position', 'y_position', 'z_position', 'current_phase']]

        t = str(t)
        dict[t] = {}
        dict[t]['x_position'] = df_cells['x_position'].tolist()
        dict[t]['y_position'] = df_cells['y_position'].tolist()
        dict[t]['z_position'] = df_cells['z_position'].tolist()
        dict[t]['current_phase'] = df_cells['current_phase'].tolist()
        
    save_compressed_json(dict, output_filename)

    return dict

def save_compressed_json(data, filename):
    with gzip.GzipFile(filename+'.gz', 'w') as f:
        f.write(json.dumps(data).encode('utf-8'))