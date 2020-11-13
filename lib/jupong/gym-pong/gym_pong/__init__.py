from gym.envs.registration import register
import numpy as np

games = ['pong']
obs_type = 'image'
frameskip = True
render_modes = {'human': True, 'headless': False}
#image_sizes = {'small':0.2, 'normal':1.0}
img_sizes = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.4, 0.6, 0.8, 1.0] # 12 Bildgroessen

for game in games:
    for render_mode, render_val in render_modes.items():
        for image_size in img_sizes:
            register(
                id=f'ju{game}2d-{render_mode}-{image_size}-v3',
                entry_point='gym_pong.envs:JuPong2D',
                kwargs={'game': game, 'obs_type': obs_type, 'render_screen': render_val, 'screen_scale' : image_size},
                max_episode_steps = 10000,
                nondeterministic = False,
            )