import gym, gym_pong
import argparse, sys, csv
import numpy as np
import gym_unrealcv
from baselines.common import atari_wrappers

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

from threading import Thread


parser = argparse.ArgumentParser()
parser.add_argument("--angle", type=float, default=0.0) # Kamerawinkel: 0 15 30 45 60
parser.add_argument("--system", type=str, default="Windows")
parser.add_argument("--factor", type=int, default=0)
args = parser.parse_args()

scale_factor_arr = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]      
scale_factor_ind = args.factor

game_system = args.system
env_name = f"jupong-3D-{game_system}-v0"
env = make_atari_env(env_name, num_env=1, seed=0)
env.envs[0].reset()
env.envs[0].scale_paddles(scale_factor_arr[scale_factor_ind])
env = VecFrameStack(env, n_stack=4)

save_path = f"ppo2_save/ppo2_save_cam_angle_{args.angle}_4"
model = PPO2.load(save_path, env=None)
model.set_env(env)

def process_environment(file_path, scale_factor_ind):
    reward_arr = []
    mean_reward = 0.0
    obs = env.reset()
    reward_sum = 0.0
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if reward != 0:
            reward_sum += reward[0]
        if done:
            reward_arr.append(reward_sum)
            mean_reward = np.mean(reward_arr)
            with open(file_path, 'r') as csv_file:
                csv_reader = list(csv.reader(csv_file, delimiter=','))
                reward_mean_arr = csv_reader[1]
            reward_mean_arr[scale_factor_ind] = mean_reward
            with open(file_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(scale_factor_arr)
                writer.writerow(reward_mean_arr)
            reward_sum = 0.0
            print(reward_mean_arr)

            
def init_csv_file(file_path, scale_factor_arr):
    try:
        with open(file_path, 'r') as csv_file:
            pass
    except FileNotFoundError:
        open(file_path, 'w').close()
        with open(file_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(scale_factor_arr)
            writer.writerow(np.zeros((len(scale_factor_arr), )))
        

csv_path = f"{save_path}.csv"

init_csv_file(csv_path, scale_factor_arr)
process_environment(csv_path, scale_factor_ind)



