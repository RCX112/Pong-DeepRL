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
import parameters as params
from collections import deque # queue data structure. fast appends. and pops. replay memory

import EnvPongDraft # Environment PyGame Pong Game Draft

# ToDo : class TaskManager ?
# ToDo : switching PaddleBot Controller on-fly

def main():
	dummy_params = params.ParameterSet({})

	PongGame = EnvPongDraft.EnvPong()

	paddle_player = PongGame.paddles[0]
	paddle_bot = PongGame.paddles[1]

	# different controllers to test each different task setting
	paddle_learner_catch_controller = EnvPongDraft.PaddleBot_Controller(PongGame, paddle_player, dummy_params)
	paddle_learner_avoid_controller = EnvPongDraft.PaddleBotAvoid_Controller(PongGame, paddle_player, dummy_params)
	paddle_bot_catch_controller = EnvPongDraft.PaddleBot_Controller(PongGame, paddle_bot, dummy_params)
	paddle_bot_avoid_controller = EnvPongDraft.PaddleBotAvoid_Controller(PongGame, paddle_bot, dummy_params)

	# testing
	# actions = [[0, 1, 0], [0, 0, 1]]
	# PongGame.step(actions)

	STEPS = 10000
	# STEPS = 2500
	STEPS_RANGE = range(STEPS)
	actions = [[0, 0, 0], [0, 0, 0]]

	# cumulative reward (pos, neg) for two players (learning agent, opponent)
	cum_reward =[[0, 0], [0, 0]]

	# cumulative hits for learner and opponent paddle
	cum_hits = [0, 0]

	# for counting hits for a task scenario for learner and opponent paddle
	interm_hits = [0, 0]

	PLAYERS = 2
	REWARD_TYPES = 2

	PLAYERS_RANGE = range(PLAYERS)
	REWARD_TYPES_RANGE = range(REWARD_TYPES)

	# TODO: simulate full task switch procedure:
	# uniform intervall [min max]
	# corresponding to what?
	# hits on the paddle for classic;
 	# ball crossing the moddle line work for both cases?
	# as for avoid case, hitting the paddle is not really helpful measure
	# TODO: in Pong Class, a counter variable for ball crossing
	# further, in detect events, rewards and punishments have to be adapted
	# depending on the task scenario
	# controller switch : can happen within or outside the env class;
	# more suitable for that is external experiment class
	# any advantage of the internal controller solution ?

	# switching is hard wired throught pre-defined SWITCH_MARKER - number of steps
	# here in example : switch each
	SWITCH_STEPS = 8
	SWITCH_MARKER = STEPS // SWITCH_STEPS

	# preparing controller to test catch scenario (classical pong)
	paddle_learner_controller = paddle_learner_catch_controller
	paddle_bot_controller = paddle_bot_catch_controller

	# switching first to classical pong
	catch = True

	# switching is hard wired throught pre-defined SWITCH_MARKER - number of steps

	for i in STEPS_RANGE:

		if (i != 0) and (i % SWITCH_MARKER == 0):
			print('CONTROLLER SWITCH.')
			if catch:
				print('Was catch. Switching to avoid.')
				print ('Hits in the catch episode were : ', interm_hits)
				paddle_learner_controller = paddle_learner_avoid_controller
				paddle_bot_controller = paddle_bot_avoid_controller
				catch = False
				interm_hits = [0, 0]
			elif not catch:
				print ('Was avoid. Switching to catch.')
				print ('Hits in the avoid episode were :', interm_hits)
				paddle_learner_controller = paddle_learner_catch_controller
				paddle_bot_controller = paddle_bot_catch_controller
				catch = True
				interm_hits = [0, 0]

		# random action index (0, 1, 2) for a random paddle movement
		# act_1 = random.randint(0, 2)
		# act_2 = random.randint(0, 2)

		# random action for player paddle
		# actions[0][act_1] = 1
		# random action for bot paddle
		# actions[1][act_2] = 1

		# actions[0] = paddle_bot_controller.act()
		actions[0] = paddle_learner_controller.act()
		actions[1] = paddle_bot_controller.act()

		# print actions

		# in observation, image_date from the screen is stored
		reward, hits, observation, hit_reward = PongGame.step(actions)
		# image_data = PongGame.render()

		for i,j in itertools.product(PLAYERS_RANGE, REWARD_TYPES_RANGE):
			cum_reward[i][j] += reward[i][j]

		for i in PLAYERS_RANGE:
			cum_hits[i] += hits[i]
			interm_hits[i] += hits[i]

		# intermediate reward test
		# print 'reward: ', reward

		# reset the actions
		# actions[0][act_1] = 0
		# actions[1][act_2] = 0
	print('Finished.')
	print('Cumulated reward: ', cum_reward)
	print('Cumulated hits: ', cum_hits)

	print('Class internal.')

	print('Cumulated reward : ', PongGame.total_reward)
	print('Cumulated hits : ', PongGame.total_hits)
	print('Ball crossings : ', PongGame.ball_cross_counter)

	return

if __name__ == "__main__":
    main()
