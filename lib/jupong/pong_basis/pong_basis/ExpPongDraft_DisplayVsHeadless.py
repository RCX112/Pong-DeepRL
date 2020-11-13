# ---- Pong Experiment Draft

# author: Jenia Jitsev, 2017, 2018

from __future__ import absolute_import, division, print_function, unicode_literals
# This ensures Python 2, 3 compatible code

# TESTED: both cv2 and tensorflow work (env: TensorFlowTest)
# import tensorflow as tf
# import cv2 # read in pixel data
import numpy as np
import random
import itertools
from collections import deque # queue data structure. fast appends. and pops. replay memory

import EnvPongDraft_Surface_Headless # Environment PyGame Pong Game Draft
import EnvPongDraft

def main():

	# INIT HAS TO HAVE SAME CONDITIONS FOR BOTH to compare screen!

	PongGame = EnvPongDraft_Surface_Headless.EnvPong()
	PongGame_Display = EnvPongDraft.EnvPong()

	# reset both instanses to exactly the same initial conditions
	PongGame._hard_reset()
	PongGame_Display._hard_reset()


	# testing
	# actions = [[0, 1, 0], [0, 0, 1]]
	# PongGame.step(actions)

	STEPS = 2000
	STEPS_RANGE = range(STEPS)
	# intermediate DEBUG Interval, for producing debug output at certain times
	DEBUG_INT = 500

	saving_images = True
	image_filename = 'screenshot'
	image_dirname = './saved_images/'
	ext = '.png'

	actions = [[0, 0, 0], [0, 0, 0]]
	actions_alt = [[0, 0, 0], [0, 0, 0]]

	# cumulative reward (pos, neg) for two players (learning agent, opponent)
	cum_reward =[[0, 0], [0, 0]]
	cum_reward_disp =[[0, 0], [0, 0]]

	# cumulative hits for learner and opponent paddle
	cum_hits = [0, 0]
	cum_hits_disp = [0, 0]

	PLAYERS = 2
	REWARD_TYPES = 2

	PLAYERS_RANGE = range(PLAYERS)
	REWARD_TYPES_RANGE = range(REWARD_TYPES)

	for i in STEPS_RANGE:
		act_1 = random.randint(0, 2)
		act_2 = random.randint(0, 2)
		actions[0][act_1] = 1
		actions[1][act_2] = 1

		# test for NON MATCH Scenario; plug this into step below for NON MATCH
		act_1_alt = random.randint(0, 2)
		act_2_alt = random.randint(0, 2)
		actions_alt[0][act_1_alt] = 1
		actions_alt[1][act_2_alt] = 1

		frame_num = i

		# print actions

		# in observation, image_date from the screen is stored
		reward, hits, observation = PongGame.step(actions)

		# TEST for Match Scenario :
		reward_disp, hits_disp, observation_disp, hit_reward = PongGame_Display.step(actions)

		# TEST for NON Match Scenario :
		# reward_disp, hits_disp, observation_disp = PongGame_Display.step(actions_alt)
		# image_data = PongGame.render()

		# if observation.all() != observation_disp.all():
		# 	print "MISMATCH!"

		if not np.array_equal(observation, observation_disp):
			# print "MISMATCH!"
			print('At Step {0} : MISMATCH!'.format(frame_num))
		# if np.array_equal(observation, observation_disp):
		# 	print "MATCH."

		if saving_images:
			if (frame_num % DEBUG_INT == 0):
				print('At Step {0} : saving images ...'.format(frame_num))
				fname_no_display = image_dirname + image_filename + '_pong_NO_DISPLAY_' + str(frame_num) + ext
				fname_display = image_dirname + image_filename + '_pong_DISPLAY_' + str(frame_num) + ext
				PongGame.save_image(fname_no_display)
				PongGame_Display.save_image(fname_display)
			# pygame.image.save(screen, image_filename)

		for i,j in itertools.product(PLAYERS_RANGE, REWARD_TYPES_RANGE):
			cum_reward[i][j] += reward[i][j]
			cum_reward_disp[i][j] += reward_disp[i][j]

		for i in PLAYERS_RANGE:
			cum_hits[i] += hits[i]
			cum_hits_disp[i] += hits_disp[i]

		# intermediate reward test
		# print 'reward: ', reward

		# reset the actions
		actions[0][act_1] = 0
		actions[1][act_2] = 0

		# reset the alternative actions (For Mismatch Scenario)
		actions_alt[0][act_1_alt] = 0
		actions_alt[1][act_2_alt] = 0

	print('Finished.')

	print('Headless: ')
	print('Cumulated reward: ', cum_reward)
	print('Cumulated hits: ', cum_hits)

	print

	print('Display: ')
	print('Cumulated reward: ', cum_reward_disp)
	print('Cumulated hits: ', cum_hits_disp)



	# print 'No display: ', observation
	# print 'Display: ', observation_disp

	return

if __name__ == "__main__":
    main()
