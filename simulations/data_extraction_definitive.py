import sys
import os
import random
import json
import itertools
import numpy as np
import time
sys.path.append('../')
from single_simulation import single_simu

# This script define the extraction of the whole dataset used in this study. 
# it calls the single simulation script to run the simulations after changing 
# the input parameters in the /helpers/simulation_paramers/simulation_parameters.json file

# Get the time for the simulation set execution
start_time = time.time()
start_CPU_time = time.process_time()

# Number of simulations for each set of parameters
two_D = True
output_folder = '/mnt/data/'

# Range extracted from supplementary table of https://www.frontiersin.org/articles/10.3389/fmolb.2022.836794/full
def get_values_and_intermediates(start, stop, num):
    linspace_values = np.linspace(start, stop, num)
    intermediate_values = [(linspace_values[i] + linspace_values[i + 1]) / 2 for i in range(len(linspace_values) - 1)]
    return linspace_values, intermediate_values

pulse_periods, pulse_periods_intermediates = get_values_and_intermediates(5, 800, 11)
range_tnfs, range_tnfs_intermediates = get_values_and_intermediates(0.001, 1, 11)
pulse_durations, pulse_durations_intermediates = get_values_and_intermediates(5, 200, 11)
tumor_radiuses = [50, 100, 275, 400]

# Generate combinations for original linspace values
combinations_raw = list(itertools.product(pulse_periods, range_tnfs, pulse_durations, tumor_radiuses))
combinations_first = [combo for combo in combinations_raw if combo[2] <= combo[0]]

# Generate combinations for intermediate values
combinations_intermediates_raw = list(itertools.product(pulse_periods_intermediates, range_tnfs_intermediates, pulse_durations_intermediates, tumor_radiuses))
combinations_intermediates = [combo for combo in combinations_intermediates_raw if combo[2] <= combo[0]]

# Combine both sets of combinations
combinations_total = combinations_first + combinations_intermediates

print(len(combinations_total))

# constraint added to increase the performances of the network, these simulations were including a bias.
combinations = [combo for combo in combinations_total if combo[1] > 0.1]

print(len(combinations))

root_dir = os.getcwd()
json_file_path = os.path.join('../helpers/simulation_parameters/simulation_parameters.json')

with open(json_file_path, 'r') as f:
    data = json.load(f)

simulation_n = 0

for combo in combinations:
    pulse_period = round(combo[0])
    tnf_conc = combo[1]
    pulse_duration = round(combo[2])
    tumor_radius = combo[3]
    
    # Print for debugging purposes
    #print(f'{pulse_period} {tnf_conc} {pulse_duration} {tumor_radius}')

    data['time_add_tnf']['value'] = pulse_period
    data['concentration_tnf']['value'] = tnf_conc
    data['duration_add_tnf']['value'] = pulse_duration
    data['tumor_radius']['value'] = tumor_radius

    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    # Uncomment this line to run the simulation
    CPU_time = single_simu(two_D, simulation_n, output_folder)
    simulation_n += 1

print(f'Number of simulations: {simulation_n} with {simulation_n} set of parameters')

end_time = time.time()
end_CPU_time = time.process_time()

# Writing the effective time required to a file
'''
with open('../times/times.txt', 'w') as f:
    f.write(f'Time taken for the simulation set execution: {end_time - start_time} seconds\n')
    f.write(f'CPU time taken for the simulation set execution: {end_CPU_time - start_CPU_time} seconds\n')

'''