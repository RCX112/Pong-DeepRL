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

# for task switching : each time switch is performed, draw a random number
# from interval min..max
MIN_BALL_EXCHANGE = 20
MAX_BALL_EXCHANGE = 30

# in case only one task is executed
NUM_BALL_EXCHANGE = 20

# one cycle contains a defined number of ball exchanges
CYCLES = 4

# one can generate ball_exchange numbers in advance :
# rand_list = random.sample(xrange(MIN, MAX), ELEMENTS)
# this would be thought without replacement, each number unique
# faster, and with replacement :
# numpy.random.randint(xrange(MIN, MAX), size=ELEMENTS)


# classic, avoid, switch
# TASK_MODE = 'classic'
# TASK_MODE = 'avoid'
TASK_MODE = 'switch'

# for the case to have either task alternation or random selection which one
# will be the next in cycle
# SWITCH_MODE = 'random'
SWITCH_MODE = 'alternate'

def main():
	dummy_params = params.ParameterSet({})

	PongGame = EnvPongDraft.EnvPong()

	paddle_player = PongGame.paddles[0]
	paddle_bot = PongGame.paddles[1]

	paddle_learner_catch_controller = EnvPongDraft.PaddleBot_Controller(PongGame, paddle_player, dummy_params)
	paddle_learner_avoid_controller = EnvPongDraft.PaddleBotAvoid_Controller(PongGame, paddle_player, dummy_params)
	paddle_learner_rand_controller = EnvPongDraft.PaddleBotRandom_Controller(PongGame, paddle_player, dummy_params)

	paddle_bot_catch_controller = EnvPongDraft.PaddleBot_Controller(PongGame, paddle_bot, dummy_params)
	paddle_bot_avoid_controller = EnvPongDraft.PaddleBotAvoid_Controller(PongGame, paddle_bot, dummy_params)
	paddle_bot_rand_controller = EnvPongDraft.PaddleBotRandom_Controller(PongGame, paddle_bot, dummy_params)

	# testing
	# actions = [[0, 1, 0], [0, 0, 1]]
	# PongGame.step(actions)

	# about 80 ball exchanges correspond to 10000 performing an action
	# STEPS = 10000
	# STEPS = 2500
	# STEPS_RANGE = range(STEPS)
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

	cycle_counter  = 0

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

	if TASK_MODE == 'classic':
		#
		PongGame.multi_task = False
		# setting active task (ID 0: classic, ID 1: avoid)
		PongGame.taskActive = [True, False]

		ball_exchange_counter  = 0

		# setting controllers
		paddle_learner_controller = paddle_learner_catch_controller
		paddle_bot_controller = paddle_bot_rand_controller


		print('EXECUTING CLASSIC Pong.')
		while ball_exchange_counter <= NUM_BALL_EXCHANGE:
			actions[0] = paddle_learner_controller.act()
			actions[1] = paddle_bot_controller.act()

			# Test print
			# print 'Cycle: ', ball_exchange_counter
			# print 'of : ', NUM_BALL_EXCHANGE

			# in observation, image_date from the screen is stored
			reward, hits, observation = PongGame.step(actions)
			# image_data = PongGame.render()

			for i,j in itertools.product(PLAYERS_RANGE, REWARD_TYPES_RANGE):
				cum_reward[i][j] += reward[i][j]

			for i in PLAYERS_RANGE:
				cum_hits[i] += hits[i]
				interm_hits[i] += hits[i]

			ball_exchange_counter = PongGame.ball_cross_counter


	elif TASK_MODE == 'avoid':
		#
		PongGame.multi_task = False
		# setting active task (ID 0: classic, ID 1: avoid)
		PongGame.taskActive = [False, True]

		ball_exchange_counter  = 0

		# set both controller to avoid
		paddle_learner_controller = paddle_learner_avoid_controller
		paddle_bot_controller = paddle_bot_rand_controller


		print('EXECUTING AVOID Pong.')
		while ball_exchange_counter <= NUM_BALL_EXCHANGE:
			actions[0] = paddle_learner_controller.act()
			actions[1] = paddle_bot_controller.act()

			# Test print
			# print 'Cycle: ', ball_exchange_counter
			# print 'of : ', NUM_BALL_EXCHANGE

			# in observation, image_date from the screen is stored
			reward, hits, observation = PongGame.step(actions)
			# image_data = PongGame.render()

			for i,j in itertools.product(PLAYERS_RANGE, REWARD_TYPES_RANGE):
				cum_reward[i][j] += reward[i][j]

			for i in PLAYERS_RANGE:
				cum_hits[i] += hits[i]
				interm_hits[i] += hits[i]

			ball_exchange_counter = PongGame.ball_cross_counter

	elif TASK_MODE == 'switch':
		print('EXECUTING Multi Task Pong (switching between classic and avoid).')
		#
		PongGame.multi_task = True
		# setting active task to start with (ID 0: classic, ID 1: avoid)
		# PongGame.taskActive = [True, False]
		# task switch inverts it
		PongGame.taskActive = [False, True]

		# print '-----------------'
		# how many task switches
		while cycle_counter < CYCLES:
			print('Cycle: ', cycle_counter)
			print
			# reset ball exchange counter for the task
			ball_exchange_counter  = 0
			# resetting task ball exchange counter
			PongGame.task_iter_counter = 0

			# switch task
			# PongGame.taskActive = not PongGame.taskActive
			# choosing task by inverting False/True active flags
			# (only one task at time is set as True)
			# !! TEMPORARY solution for two tasks only !!
			if SWITCH_MODE == 'alternate':
				# inverting task flags to alternate tasks
				PongGame.taskActive[:] = [not t for t in PongGame.taskActive]
			elif SWITCH_MODE == 'random':
				PongGame.taskActive = [False, False]
				idx = random.randint(0, 1)
				PongGame.taskActive[idx] = True


			# returning index of the active task (only one marked as True)
			task_id = PongGame.taskActive.index(True)

			# drawing number of ball exchange times from interval min..max
			task_ball_exchange = random.randint(MIN_BALL_EXCHANGE, MAX_BALL_EXCHANGE)
			print('Task switching.')
			print('Ball exchanges to perform: ', task_ball_exchange)

			if task_id == 0: # classic on
				print('Switching to CLASSIC.')
				print('PongGame.taskActive : ', PongGame.taskActive)

				paddle_learner_controller = paddle_learner_catch_controller
				paddle_bot_controller = paddle_bot_rand_controller

				while ball_exchange_counter <= task_ball_exchange:
					actions[0] = paddle_learner_controller.act()
					actions[1] = paddle_bot_controller.act()

					# Test print
					# print 'Cycle: ', ball_exchange_counter
					# print 'of : ', NUM_BALL_EXCHANGE

					# in observation, image_date from the screen is stored
					reward, hits, observation = PongGame.step(actions)
					# image_data = PongGame.render()

					for i,j in itertools.product(PLAYERS_RANGE, REWARD_TYPES_RANGE):
						cum_reward[i][j] += reward[i][j]

					for i in PLAYERS_RANGE:
						cum_hits[i] += hits[i]
						interm_hits[i] += hits[i]

					ball_exchange_counter = PongGame.task_iter_counter
			elif task_id == 1: # avoid on
				print('Switching to AVOID.')
				print('PongGame.taskActive : ', PongGame.taskActive)

				paddle_learner_controller = paddle_learner_avoid_controller
				paddle_bot_controller = paddle_bot_catch_controller

				while ball_exchange_counter <= task_ball_exchange:
					actions[0] = paddle_learner_controller.act()
					actions[1] = paddle_bot_controller.act()

					# Test print
					# print 'Cycle: ', ball_exchange_counter
					# print 'of : ', NUM_BALL_EXCHANGE

					# in observation, image_date from the screen is stored
					reward, hits, observation = PongGame.step(actions)
					# image_data = PongGame.render()

					for i,j in itertools.product(PLAYERS_RANGE, REWARD_TYPES_RANGE):
						cum_reward[i][j] += reward[i][j]

					for i in PLAYERS_RANGE:
						cum_hits[i] += hits[i]
						interm_hits[i] += hits[i]

					ball_exchange_counter = PongGame.task_iter_counter

			cycle_counter += 1

	print('---------------')
	print('---------------')

	print('Finished.')
	print('Cumulated reward: ', cum_reward)
	print('Cumulated hits: ', cum_hits)

	print('Class internal.')

	print('Cumulated reward : ', PongGame.total_reward)
	print('Cumulated hits : ', PongGame.total_hits)
	print('Ball crossings : ', PongGame.ball_cross_counter)
	print('Total ball crossings : ', sum(PongGame.ball_cross_counter))

	return

if __name__ == "__main__":
    main()
