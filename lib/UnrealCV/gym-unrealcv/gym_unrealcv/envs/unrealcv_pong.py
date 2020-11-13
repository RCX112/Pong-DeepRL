import gym
import unrealcv
import numpy as np
import matplotlib.pyplot as plt
from gym_unrealcv.envs.utils import env_unreal, unrealcv_basic
from io import BytesIO
from gym.utils import seeding

class JuPong3D(gym.Env):
    def __init__(self,
                ENV_BIN = "RealisticRendering_Win64/WindowsNoEditor/RealisticRendering/Binaries/Win64/RealisticRendering.exe",
                docker = False,
                resolution=(640, 480, 4),
                frameskip=(2, 5)
    ):
        self.PERC = (25/160, 5/120)
        self.DARK_LIMIT = 70
        self.STD_SHAPE = (110, 110, 4)
        
        self.cam_id = 0
        self.docker = docker
        self.frameskip = frameskip
        self.resolution = resolution
        self._action_set = [0, 1]
        self.ale = self
        self.np_random = seeding.np_random(0)[0]
        
        self.unreal = env_unreal.RunUnreal(ENV_BIN=ENV_BIN)
        env_ip, env_port = self.unreal.start(docker, resolution)
        print(f"Port lautet: {env_port}") # Standard Port: 9000 in unrealcv.ini von ENV_BIN

        self.ACTION_LIST = [0, 1]
        self.count_steps = 0
        self.MAX_STEPS = 10000
        self.action_space = gym.spaces.Discrete(len(self.ACTION_LIST))
        self.prev_ob = np.zeros((110, 110))
        self.cur_ob = np.zeros((110, 110))
        self.return_val = 0.0

        self.unrealcv = unrealcv_basic.UnrealCv(port=env_port, ip=env_ip, env=self.unreal.path2env, cam_id=self.cam_id, resolution=resolution)
        self.unreal_client = self.unrealcv.client
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.STD_SHAPE, dtype=np.uint8)

    def _step(self, action):
        reward = 0.0
        self.count_steps += 1
        done = False
        
        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = np.random.randint(self.frameskip[0], self.frameskip[1])
        
        for _ in range(num_steps):
            if _ != (num_steps-1):
                self.unpause_game()
                rew, done = self.act(action, 1)
            else:
                self.unpause_game()
                rew, done = self.act(action, 1)
            reward += rew
            
        state = self.get_state()
        self.return_val += reward
        
        return state, reward, done, {}
        
    def prepro(self, I):
        lowl = int(self.PERC[1] * self.resolution[1])
        lowr = int(self.PERC[0] * self.resolution[0])
        I = np.array(I[lowl:(self.resolution[1]-lowl):, lowr:(self.resolution[0]-lowr):, :])
        
        #print(I.shape)
        
        return I

    def _seed(self, seed=None):
        return seed
    
    def _reset(self):
        self.unreal_client.request("vset /agent/reset")
        self.pause_game()
        self.prev_ob = np.zeros((110, 110))
        self.cur_ob = np.zeros((110, 110))
        return self.get_state()

    def _close(self):
        pass
       
    def move_up(self, pause_game):
        return self.unreal_client.request(f"vset /agent/moveup/{pause_game}")
       
    def move_down(self, pause_game):
        return self.unreal_client.request(f"vset /agent/movedown/{pause_game}")
       
    def act(self, action, pause_game):
        if action == 0:
            res = self.move_up(pause_game)        
        elif action == 1:
            res = self.move_down(pause_game)

        (reward, episode_over) = float(res.split(',')[0]), bool(int(res.split(',')[1]))

        return reward, episode_over
   
    def get_state(self, cam_id = 0, viewmode = 'lit'):
        #return self.get_diff_image(cam_id, viewmode)
        return self.get_standard_image(cam_id, viewmode)
        #return self.get_transposed_image(cam_id, viewmode)
       
    def unpause_game(self):
        self.unreal_client.request("vset /agent/unpause")
       
    def pause_game(self):
        self.unreal_client.request("vset /agent/pause")
       
    def change_camera_angle(self, angle, cam_id = 0):
        self.unreal_client.request(f'vset /camera/{cam_id}/rotation 0 {angle} 0')
        loc_x = -3100.0
        loc_y = -3100.0 * np.tan(np.deg2rad(angle))
        loc_z = 1900.0
        self.unreal_client.request(f'vset /camera/{cam_id}/location {loc_x} {loc_y} {loc_z}')
       
    def get_diff_image(self, cam_id, viewmode):
        self.prev_ob = self.cur_ob
        self.cur_ob = self.prepro(self.unrealcv.decode_png(self.unreal_client.request(f'vget /camera/{cam_id}/{viewmode} png')))
        diff = self.cur_ob - self.prev_ob
        
        return diff
       
    def get_transposed_image(self, cam_id, viewmode):
        return self.unrealcv.decode_png(self.unreal_client.request(f'vget /camera/{cam_id}/{viewmode} png')).transpose(1, 0, 2)
       
    def get_standard_image(self, cam_id, viewmode):
        return self.prepro(self.unrealcv.decode_png(self.unreal_client.request(f'vget /camera/{cam_id}/{viewmode} png')))
        
    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]
        
    def lives(self):
        return 1
    
    def scale_paddles(self, scale_factor):
        self.unreal_client.request(f"vset /agent/scale/{scale_factor}")
        
ACTION_MEANING = {
    0: "UP",
    1: "DOWN"
}
"""
ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
"""