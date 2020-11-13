import gym
import unrealcv
import numpy as np
import math, random
import matplotlib.pyplot as plt
from gym_unrealcv.envs.utils import env_unreal, unrealcv_basic
import time

ENV_BIN = "RealisticRendering_Win64/WindowsNoEditor/RealisticRendering/Binaries/Win64/RealisticRendering.exe"
docker = False
resolution=(640, 480, 4)
cam_id = 0

unreal = env_unreal.RunUnreal(ENV_BIN=ENV_BIN)
env_ip, env_port = unreal.start(docker, resolution)

ACTION_LIST = [0, 1]
count_steps = 0
max_steps = 1000
action_space = gym.spaces.Discrete(len(ACTION_LIST))

unrealcv = unrealcv_basic.UnrealCv(port=env_port, ip=env_ip, env=unreal.path2env, cam_id=cam_id, resolution=resolution)
state = unrealcv.read_image(cam_id, 'lit')
observation_space = gym.spaces.Box(low=0, high=255, shape=(80, 80, 3), dtype=np.uint8)

unreal_client = unrealcv.client
unreal_client.request("vset /agent/reset")
while True:
    count_steps += 1
    k = random.randint(0,1)
    if k == 0:
        res = unreal_client.request("vset /agent/moveup")
    else:
        res = unreal_client.request("vset /agent/movedown")
        
    (reward, episode_over) = float(res.split(',')[0]), bool(int(res.split(',')[1]))
    if(reward != 0.0):
        print(count_steps, k, res, episode_over)
    #state = unrealcv.read_image(0, 'lit')
    #print(state.shape)
    #plt.imshow(state)
    #plt.show()
   