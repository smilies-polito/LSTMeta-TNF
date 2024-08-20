import sys
import os
sys.path.append('../')
from interface.interface import interface
from data_conversion.json_conversions import *

# This script defines the execution of a single PhysiBoSS simulation with the given parameters.

def single_simu(two_D, simulation, output_folder, root_dir):
    
    json_file_path = os.path.join(root_dir, 'helpers/simulation_parameters/simulation_parameters.json')

    folders = ['cell_data', 'input_parameters']

    # Creazione delle cartelle se non esistono già
    for folder in folders:
        folder_path = os.path.join(output_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # define interface object
    my_interface = interface(root_dir, json_file_path, two_D)
    
    my_interface.update_parameters()

    # Execute the simulation with new parameters

    output_folder_sim, CPU_time = my_interface.execute_simulation(simulation)

    # Convert the output data to json files
    #microenv_dictionary = microenv_data(simulation, output_folder)
    cell_dictionary = cell_data(simulation, output_folder)

    input_parameters_save(simulation, output_folder, json_file_path)
    
    return CPU_time

if __name__ == "__main__":

    two_D = False
    single_simu(two_D, simulation=0)
