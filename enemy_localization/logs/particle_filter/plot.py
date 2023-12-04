import os
import re
import matplotlib.pyplot as plt
import pandas as pd

dirname = '01'
directory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dirname)   # This would be the directory containing the log files


# Function to extract numbers from a string
def extract_numbers(s):
    return [int(num) for num in re.findall(r'\d+', s)]

enemies = set()
agents =set()

def load_data_from_directory():

    # Dictionary to hold the data
    data = {
        'true_distances': {},
        'estimated_distances': {},
        'noisy_distances': {},
        'true_positions': {},
        'estimated_positions': {}
    }
    
    # Regex patterns to identify the file types and indices
    true_dist_pattern = re.compile(r'true_distances_(\d+)\.log$')
    estimated_dist_pattern = re.compile(r'estimated_distances_agent_(\d+)\.log$')
    noisy_dist_pattern = re.compile(r'noisy_distances_(\d+)\.log$')
    true_pos_pattern = re.compile(r'true_positions_enemy_(\d+)\.log$')
    estimated_pos_pattern = re.compile(r'estimated_positions_enemy_(\d+)\.log$')

    # Read files from the directory
    for filename in os.listdir(directory_path):
        if not filename.endswith('log'):
            continue

        file_path = os.path.join(directory_path, filename)
        
        with open(file_path, 'r') as file:
            parsed_data = [extract_numbers(line) for line in file.readlines()]
        
        # Check and load true distances
        match = true_dist_pattern.search(filename)
        if match:
            agent_index = int(match.group(1))
            agents.add(agent_index)
            data['true_distances'][agent_index] = pd.DataFrame(parsed_data, 
                columns=[f"True Distance to Enemy {(agent_index + (1 + 2*i)) % 4}" for i in range(2)])
            continue

        # Check and load estimated distances
        match = estimated_dist_pattern.search(filename)
        if match:
            agent_index = int(match.group(1))
            agents.add(agent_index)
            data['estimated_distances'][agent_index] = pd.DataFrame(parsed_data,
                columns=[f"Estimated Distance to Enemy {(agent_index + (1 + 2*i)) % 4}" for i in range(2)])
            continue
        
        # Check and load noisy distances
        match = noisy_dist_pattern.search(filename)
        if match:
            agent_index = int(match.group(1))
            agents.add(agent_index)
            data['noisy_distances'][agent_index] = pd.DataFrame(parsed_data,
                columns=[f"Noisy Distance to Enemy {(agent_index + (1 + 2*i)) % 4}" for i in range(2)])
            continue

        # Check and load true positions
        match = true_pos_pattern.search(filename)
        if match:
            enemy_index = int(match.group(1))
            enemies.add(enemy_index)
            data['true_positions'][enemy_index] = pd.DataFrame(parsed_data,
                columns=["X", "Y"])
            continue

        # Check and load estimated positions
        match = estimated_pos_pattern.search(filename)
        if match:
            enemy_index = int(match.group(1))
            enemies.add(enemy_index)
            data['estimated_positions'][enemy_index] = pd.DataFrame(parsed_data,
                columns=["X", "Y"])
            continue
    
    return data

# Function to plot distances for a specific agent in a grid
def plot_distances_for_agent(agent_index, data, axs):
    
    true_distances = data['true_distances'][agent_index]
    estimated_distances = data['estimated_distances'][agent_index]
    noisy_distances = data['noisy_distances'][agent_index]

    # Adjust the loop for the number of enemies you have
    for i, enemy in enumerate(sorted(enemies)):
        # Plotting true, estimated, and noisy distances to each enemy in a subplot
        axs[i].plot(true_distances[f"True Distance to Enemy {enemy}"], label=f"True Distance to Enemy {enemy}", color="green", linestyle="--")
        axs[i].plot(estimated_distances[f"Estimated Distance to Enemy {enemy}"], label=f"Estimated Distance to Enemy {enemy}", color="blue")
        axs[i].plot(noisy_distances[f"Noisy Distance to Enemy {enemy}"], label=f"Noisy Distance to Enemy {enemy}", color="red", alpha=0.7)
        axs[i].set_title(f"Agent {agent_index}: Distance Comparison to Enemy {enemy}")
        axs[i].set_xlabel("Turn")
        axs[i].set_ylabel("Distance")
        axs[i].legend(loc='upper right')
        axs[i].grid(True)

        # Calculating and displaying stats
        mse_estimated = ((true_distances[f"True Distance to Enemy 1"] - estimated_distances[f"Estimated Distance to Enemy 1"])**2).mean()
        mse_noisy = ((true_distances[f"True Distance to Enemy 1"] - noisy_distances[f"Noisy Distance to Enemy 1"])**2).mean()
        var_estimated_error = (true_distances[f"True Distance to Enemy 1"] - estimated_distances[f"Estimated Distance to Enemy 1"]).var()
        var_noisy_error = (true_distances[f"True Distance to Enemy 1"] - noisy_distances[f"Noisy Distance to Enemy 1"]).var()

        stats_text = f"MSE (Estimated to True): {mse_estimated:.2f}\n" \
                f"MSE (Noisy to True): {mse_noisy:.2f}\n" \
                f"Variance (Estimated to True): {var_estimated_error:.2f}\n" \
                f"Variance (Noisy to True): {var_noisy_error:.2f}"
        
        axs[i].text(0.2, 0.85, stats_text, horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))


def plot_distance_evaluations():
    data = load_data_from_directory()



    fig, axs = plt.subplots(len(agents), len(enemies), figsize=(16, 16))  
    for i, agent in enumerate(sorted(agents)):
        plot_distances_for_agent(agent, data, axs[i])  # Plotting for agent 0 as an example
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, bottom=0.1, top=0.9)
    plt.savefig(os.path.join(directory_path, 'distance_evaluations.png'))
    plt.show()



def print_mean_manhattan_distance_error():
    data = load_data_from_directory()
    for enemy in sorted(enemies):
        mean_manhattan_distance_error = (data['true_positions'][enemy] - data['estimated_positions'][enemy]).abs().sum(axis=1).mean()
        print(f'Mean Manhattan Distance Error for Enemy {enemy}: {mean_manhattan_distance_error:.2f}')

    for agent in sorted(data['true_distances'].keys()):
        true_distances = data['true_distances'][agent].dropna()
        true_distances.columns=['enemy1','enemy2']
        estimated_distances = data['estimated_distances'][agent].dropna()
        estimated_distances.columns=['enemy1','enemy2']
        noisy_distances = data['noisy_distances'][agent].dropna()
        noisy_distances.columns=['enemy1','enemy2']
        manhattan_estimated = ((true_distances - estimated_distances)).abs().mean()
        manhattan_noisy = ((true_distances - noisy_distances)).abs().mean()
        print(f'Mean Estimated Manhattan Distance Error for Agent {agent}: {manhattan_estimated.sum():.2f}')
        print(f'Mean Noisy Manhattan Distance Error for Enemy {enemy}: {manhattan_noisy.sum():.2f}')
# 18 8 tru
#21 13 esti


print_mean_manhattan_distance_error()
plot_distance_evaluations()


# TODO add table which has mean_manhattan_distance_error / mse / var
# as well as other stats which are currently in the plots
# table should be in last row and span all columns