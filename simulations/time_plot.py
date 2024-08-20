import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Tis script is used to create a plot comparing the time of the simulations with the metamodel.
# the time of the metamodel is constant and equal to 0.008s (set manually).

# Function to adjust the brightness of a color
def adjust_brightness(color, factor):
    # Convert color to RGB
    r, g, b = mcolors.to_rgb(color)
    # Clamp values after adjustment to be between 0 and 1
    r = min(max(0, r * factor), 1)
    g = min(max(0, g * factor), 1)
    b = min(max(0, b * factor), 1)
    # Return the adjusted color
    return mcolors.to_hex((r, g, b))

# Function to create and save the plot
def create_plot(times_df, time_key, title, filename):
    # create a vector for all the different radii
    time_radius_50 = times_df[times_df['tumor_radius'] == 50][time_key]
    time_radius_100 = times_df[times_df['tumor_radius'] == 100][time_key]
    time_radius_275 = times_df[times_df['tumor_radius'] == 275][time_key]
    time_radius_400 = times_df[times_df['tumor_radius'] == 400][time_key]

    # create time vectors simulating the execution of subsequent simulations
    stop = 5000

    vector_radius_50 = np.arange(0, stop, time_radius_50.iloc[0])[:75]
    vector_radius_100 = np.arange(0, stop, time_radius_100.iloc[0])[:75]
    vector_radius_275 = np.arange(0, stop, time_radius_275.iloc[0])[:75]
    vector_radius_400 = np.arange(0, stop, time_radius_400.iloc[0])[:75]

    vector_metamodel_y = np.arange(180, stop, 0.008)[:75]

    # Create a new figure with a specific size
    plt.figure(figsize=(9, 9))

    # Base red color
    base_color = '#F55733'

    # Adjust the brightness for different vectors
    color_radius_50 = adjust_brightness(base_color, 0.6)  # Slightly darker
    color_radius_100 = base_color  # Original color
    color_radius_275 = adjust_brightness(base_color, 1.4)  # Slightly lighter
    color_radius_400 = adjust_brightness(base_color, 1.8)  # Even lighter

    # Plot the lines
    plt.plot(vector_radius_50, label='Radius 50', color=color_radius_50, linewidth=5)
    plt.plot(vector_radius_100, label='Radius 100', color=color_radius_100, linewidth=5)
    plt.plot(vector_radius_275, label='Radius 275', color=color_radius_275, linewidth=5)
    plt.plot(vector_radius_400, label='Radius 400', color=color_radius_400, linewidth=5)
    plt.plot(vector_metamodel_y, label='Metamodel', color='#323B4B', linewidth=5)

    # Function to find the first intersection or exceeding point
    def find_first_exceeding_point(vector, metamodel_vector):
        for i in range(len(vector)):
            if vector[i] > metamodel_vector[i]:
                return i
        return -1

    # Find the first exceeding points
    exceeding_points = [
        find_first_exceeding_point(vector_radius_50, vector_metamodel_y),
        find_first_exceeding_point(vector_radius_100, vector_metamodel_y),
        find_first_exceeding_point(vector_radius_275, vector_metamodel_y),
        find_first_exceeding_point(vector_radius_400, vector_metamodel_y),
    ]

    # Plot the vertical dashed lines
    for point in exceeding_points:
        if point != -1:
            plt.plot([point, point], [0, vector_metamodel_y[point]], color='grey', linestyle='--', linewidth=2, alpha=0.5)

    # Set the axis labels and their font sizes
    plt.xlabel('Simulations', fontsize=26)
    plt.ylabel('Time (s)', fontsize=26)

    # Set the title of the graph with plt.suptitle() and its font size
    plt.suptitle(title, fontsize=30)

    # Set the font size for the axis ticks
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.yscale('log')

    # Add a legend with a specific font size
    plt.legend(fontsize=22)

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(filename)

# Read the CSV file
if __name__ == '__main__':
    time_filename = '../times/mean_times.csv'
    times_df = pd.read_csv(time_filename, sep='\t')

    # Create and save the plots
    # create_plot(times_df, 'mean_time', 'Time Comparison', '../times/time_plot.png') # plot with execution time
    create_plot(times_df, 'mean_time_CPU', 'Computation Time Comparison', '../times/cpu_time_plot.png') #plot with CPU/GPU time
