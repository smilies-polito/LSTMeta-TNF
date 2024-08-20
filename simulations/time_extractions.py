import sys
import os
import shutil
import json
import numpy as np
import pandas as pd
import time
sys.path.append('../')
from single_simulation import single_simu

# This script is used to extract the mean time of the simulation for different tumor radiuses

#number of simulations for each set of parameters
two_D = True
output_folder = '/mnt/tmp/'

#range extracted from supplementary table of https://www.frontiersin.org/articles/10.3389/fmolb.2022.836794/full
tumor_radiuses = [50, 100, 275, 400]

num_equal_smulations = 4

os.chdir('..')
root_dir = os.getcwd()
json_file_path = os.path.join(root_dir, 'helpers/simulation_parameters/simulation_parameters.json')

with open(json_file_path, 'r') as f:
    data = json.load(f)

#create the dataframe to store all the data
df = pd.DataFrame(columns=["tumor_radius", "mean_time", "mean_time_CPU"])

simulation_n = 1

for tumor_radius in tumor_radiuses:

    pulse_period = 5
    tnf_conc = 0.001
    pulse_duration = 5
    
    print(f'{pulse_period} {tnf_conc} {pulse_duration} {tumor_radius}')

    data['time_add_tnf']['value'] = pulse_period
    data['concentration_tnf']['value'] = tnf_conc
    data['duration_add_tnf']['value'] = pulse_duration
    data['tumor_radius']['value'] = tumor_radius

    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    delta_times = []
    delta_times_CPU = []

    for i in range(num_equal_smulations):

        #compute the times and store th mean
        start_time = time.time()
        start_CPU_time = time.process_time()

        CPU_sub_time = single_simu(two_D, simulation_n, output_folder, root_dir)


        end_time = time.time()
        end_CPU_time = time.process_time()

        delta_times.append(end_time - start_time)
        delta_times_CPU.append((end_CPU_time - start_CPU_time) + CPU_sub_time)
        simulation_n += 1
    
    #compute the mean and store it in a dataframe
    mean_time = np.mean(delta_times)
    mean_time_CPU = np.mean(delta_times_CPU)

    # New data to be added
    new_data = {"tumor_radius": tumor_radius, "mean_time": mean_time, "mean_time_CPU": mean_time_CPU}

    # Convert the new data into a DataFrame
    new_row = pd.DataFrame([new_data])

    # Use pd.concat to add the new row to the existing dataframe
    df = pd.concat([df, new_row], ignore_index=True)

#remove the tmp folder since the dataare not usefull
shutil.rmtree(output_folder)

df.to_csv("../times/mean_times.csv", index=False, sep = '\t')