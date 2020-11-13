"""
The class 'JuPong2D_PPO_Plot' of this file plots the results of the RLlib-training in matplotlib.
Here the collected return-values of given parameter sets and neuronal networks will be shown in a scatter plot.

"""
import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse
import os

class JuPong2D_PPO_Plot:
    """
    The class 'JuPong2D_PPO_Plot' of this file plots the results of the RLlib-training in matplotlib.
    """
    def __init__(self, output):
        """
        Constructor of the class 'JuPong2D_PPO_Plot'. All the subfolders of the RLlib-results-folder will be saved in an
        array.
        :param output: Path to the results folder
        """
        self.output = output
        self.folders = [f.path for f in os.scandir(self.output) if f.is_dir() and not f.name.startswith(".")]
        
    def plot_paddle_length(self):
        """
        Plots the results of the analysis of the parameter paddle length by plotting all the collected mean
        return-values with their standard deviation.
        """
        pl_folders = [folder for folder in self.folders if folder.split("/")[-1].split("_")[-2] == "PaddleLength"]

        fig = plt.figure()
        for pl_folder in pl_folders:
            pl_factor = float(pl_folder.split("_")[-1])
            reward_matrix = []
            scale_factor_matrix = []
            sessions = [f.path for f in os.scandir(pl_folder) if f.is_dir() and f.name.startswith("session")]
            self.handle_sessions(sessions, reward_matrix, scale_factor_matrix)
            mean_rewards = np.mean(reward_matrix, axis = 0)
            std_vals = np.std(reward_matrix, axis = 0)
            scale_factors = np.mean(scale_factor_matrix, axis=0)
            std_rewards_up = mean_rewards + std_vals
            std_rewards_down = mean_rewards - std_vals
            plt.vlines(scale_factors, std_rewards_down, std_rewards_up, zorder = -1, linestyle="dashed")
            plt.scatter(scale_factors, mean_rewards, label = pl_factor, s = 60)
        plt.xlabel("Factor of the paddle-length", fontsize=20)
        plt.ylabel("Mean Return-Values\n with std. deviation", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(loc = "lower right", fontsize=12)
        fig.savefig(f"{self.output}/paddle_length.pdf", format="pdf", bbox_inches='tight', dpi=800, pad_inches=0.1)
        plt.show()

    def handle_sessions(self, sessions, reward_matrix, scale_factor_matrix):
        """
        Paddle Length Sessions
        """
        for session in sessions:
            for pl_file in os.listdir(session):
                if pl_file.endswith(".csv") and pl_file.startswith("paddle_length"):
                    with open(f"{session}/{pl_file}") as csv_file:
                        rewards = []
                        scale_factors = []
                        csv_reader = list(csv.reader(csv_file, delimiter=','))
                        for i, reward in enumerate(csv_reader[1]):
                            rewards.append(np.round(float(reward), 2))
                            scale_factors.append(float(csv_reader[0][i]))
                        reward_matrix.append(rewards)
                        scale_factor_matrix.append(scale_factors)
                    break


def start():
    """
    Start-method of the RLlib-plotting-library.
    Usage: plot_rllib [-h] output

    positional arguments:
        output      Path to the results folder.

    optional arguments:
        -h, --help  show this help message and exit
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help='Path to the results folder.')
    args = parser.parse_args()

    ploter = JuPong2D_PPO_Plot(args.output)
    ploter.plot_paddle_length()
    print(f"Successfully saved the results in the folder '{args.output}'")


if __name__ == "__main__":
    start()
