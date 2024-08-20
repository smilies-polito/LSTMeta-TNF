import os
import subprocess
import xml.etree.ElementTree as ET
import shutil
import json
import sys
import time
import resource
import matplotlib.pyplot as plt
from pctk import multicellds
sys.path.append('../')

# This script is the interface between the user and the PhysiBoSS model.
# It allows to update the parameters of the model,
#  execute the simulation and extract the data from the simulation.

class interface:
    def __init__(self, root_dir, json_file_path, two_D):

        # define the directory
        self.root_dir = root_dir
        self.PhysiBoSS_dir = '/tmp/model'
        self.two_D = two_D

        #define the parameter dict
        self.json_file_path = json_file_path
        with open(json_file_path, 'r') as file:
            self.parameter_dict = json.load(file)
        
        if self.two_D:
            self.parameter_dict['z_min']['value'] = -10
            self.parameter_dict['z_max']['value'] = 10
            self.parameter_dict['use_2D']['value'] = 'true'
        else:
            self.parameter_dict['z_min']['value'] = -500
            self.parameter_dict['z_max']['value'] = 500
            self.parameter_dict['use_2D']['value'] = 'false'

        self.output_folder = os.path.join(self.PhysiBoSS_dir, 'output')

    def update_parameters(self):
        
        # File path for the physicell settings
        physicell_setting_file = os.path.join(self.PhysiBoSS_dir, 'config/PhysiCell_settings.xml')

        # Path for the directory with the saving of old simulation
        old_simu_path = os.path.join(self.PhysiBoSS_dir, 'starting_file_trial')
        
        # Upload XML file
        tree = ET.parse(physicell_setting_file)
        root = tree.getroot()
        
        # Extract list of parameters to update
        parameters = list(self.parameter_dict.keys())
        
        for param in parameters:
            # Find parameter tags
            tags = self.parameter_dict[param]['path'].split('/')
            
            # Find tag to modify
            element = root
            for tag in tags:
                if element is not None:
                    if tag == 'variable' and 'name' in self.parameter_dict[param]:
                        # Find the variable with the specific name
                        found = False
                        for var in element.findall(tag):
                            if var.attrib['name'] == self.parameter_dict[param]['name']:
                                element = var
                                found = True
                                break
                        if not found:
                            element = None
                            break
                    elif tag != 'variable':
                        element = element.find(tag)
                
            #Update the value
            new_value = str(self.parameter_dict[param]['value'])

            if element is not None:
                element.text = new_value

            # Save XML file
            tree.write(physicell_setting_file)
            
        return 'Settings updated succesfully!'
    
    def execute_simulation(self, iteration):
        # Change current working directory
        os.chdir(self.PhysiBoSS_dir)
        CPU_time = 0
        # Need to be updated but for now let's mantain this
        executable_file = 'tnf-cancer-model'
        
        # Recreate output folder
        first_make_command = ["make", 'data-cleanup']
        usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        subprocess.run(first_make_command, check=True, stdout=subprocess.DEVNULL)
        usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)

        CPU_time = CPU_time + (usage_end.ru_utime - usage_start.ru_utime)
        # Cleanup the projects
        # Delete .o files
        if iteration == 0:
            
            clean_make_command = ["make", 'clean']
            usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
            subprocess.run(clean_make_command, check=True, stdout=subprocess.DEVNULL)
            usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
            CPU_time = CPU_time + (usage_end.ru_utime - usage_start.ru_utime)
        
            make_command = ["make"]
            usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
            subprocess.run(make_command, check=True, stdout=subprocess.DEVNULL)
            usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
            CPU_time = CPU_time + (usage_end.ru_utime - usage_start.ru_utime)
        
        # Run the simulation
        execute_command = ["./" + executable_file]
        usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        subprocess.run(execute_command, check=True, stdout=subprocess.DEVNULL) 
        usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)

        CPU_time = CPU_time + ((usage_end.ru_utime - usage_start.ru_utime)/8)
        
        return self.output_folder, CPU_time
    
    def alive_cells(self):
        # Creating a MCDS reader
        reader = multicellds.MultiCellDS(output_folder=self.output_folder)

        # Creating an iterator to load a cell DataFrame for each stored simulation time step
        df_iterator = reader.cells_as_frames_iterator()

        step_alive = []
        step_apoptotic = []
        step_necrotic = []
        time_steps = []
        print("\n")

        for (t, df_cells) in df_iterator:
            alive = (df_cells.current_phase==14).sum()
            apoptotic = (df_cells.current_phase==100).sum()
            necrotic = (df_cells.current_phase==101).sum()
            step_alive.append(alive)
            step_apoptotic.append(apoptotic)
            step_necrotic.append(necrotic)
            time_steps.append(t)
            print(f"Total alive {alive}, necrotic {necrotic} and apoptotic {apoptotic} cells at time {t}")

        
        return time_steps, step_alive, step_apoptotic, step_necrotic

    
    def plot(self, time_steps, step_alive, step_necrotic, step_apoptotic):
    # Colors definition
        color_alive = (0.25, 1, 0.25)  # Green
        color_necrotic = (0.6, 0.3, 0.1)  # Brown
        color_apoptotic = (1, 0, 0)  # Red
        
        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, step_alive, label='Alive Cells', color=color_alive)
        plt.plot(time_steps, step_necrotic, label='Necrotic Cells', color=color_necrotic)
        plt.plot(time_steps, step_apoptotic, label='Apoptotic Cells', color=color_apoptotic)
        plt.xlabel('Time (min)')
        plt.ylabel('Number of Cells')
        plt.title('Cell Population over Time')
        plt.xticks([0, 300, 600, 900, 1200], labels=['0', '300', '600', '900', '1200'])
        if self.two_D :
            plt.yticks([0, 100, 200, 300], labels=['0', '100', '200', '300'])
        else:
            plt.yticks([0, 1000, 2000, 3000], labels=['0', '1000', '2000', '3000'])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_folder, 'cell_population_over_time.png'))
        plt.close()

    def resistance_detection(self, alive_cells):

        
        stable_states = ['TNF TNFR RIP1 RIP1ub RIP1K IKK NFkB BCL2 ATP cIAP XIAP cFLIP Survival',
                        'FASL TNF TNFR RIP1 RIP1ub RIP1K IKK NFkB BCL2 ATP cIAP XIAP cFLIP Survival',
                        'TNF TNFR DISC-TNF FADD RIP1 RIP1ub RIP1K IKK NFkB BCL2 ATP cIAP XIAP cFLIP Survival',
                        'FASL DISC-FAS FADD RIP1 RIP1ub RIP1K IKK NFkB BCL2 ATP cIAP XIAP cFLIP Survival',
                        'FASL TNF TNFR DISC-TNF DISC-TNF FADD RIP1 RIP1ub RIP1K IKK NFkB BCL2 ATP cIAP XIAP cFLIP Survival']
        for i in range(len(stable_states)):
            stable_states[i] = stable_states[i].split(' ')

        # read the bool_data.txt file
        filename = os.path.join(self.PhysiBoSS_dir, 'starting_file_trial/bool_data.txt')
        command = ["awk", "{print}", filename]
        result = subprocess.run(command, capture_output=True, text=True)
        lines = result.stdout.split('\n')[:-1]

        counter_stable = 0
        for raw_line in lines:
            raw_line = raw_line.split(' ')
            line = []
            for i in range(len(raw_line)-1):
                flag = raw_line[i].replace(";", "").split('=')
                node = flag[0]
                value = flag[1]

                if value == '1':
                    line.append(node)

            for state in stable_states:
                if set(state).issubset(set(line)):
                    counter_stable += 1
                    break  # exit the inner loop if a match is found

        percentage_of_resistant = counter_stable/alive_cells
        
        return percentage_of_resistant, counter_stable



