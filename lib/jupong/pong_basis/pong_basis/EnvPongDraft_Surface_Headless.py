#
# Simple Pong Game Environment Draft based upon PyGame
# Environment Pong Game
#

# author: Jenia Jitsev, 2017, 2018

from __future__ import absolute_import, division, print_function, unicode_literals
# This ensures Python 2, 3 compatible code
from pong_basis import EnvPongDraft
import os

import pygame
import random

import numpy as np
import parameters as params

from pygame.locals import *

# size of the game screen (in case it is square)
SCALING = 1.0
#SCALING = 0.2
SCREEN_SIZE = int(400 * SCALING)

# size of the game screen
WINDOW_WIDTH = SCREEN_SIZE
WINDOW_HEIGHT = SCREEN_SIZE

# RGB colors for the paddle and ball
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# size of the paddle
PADDLE_WIDTH = int(10 * SCALING)
PADDLE_HEIGHT = int(60 * SCALING)

# distance from the edge of the window
PADDLE_BUFFER = 10  # 15 used in simplified state version

PADDLE_COLOR = WHITE
PADDLE_SHAPE = 'rect'

# size of the ball
BALL_WIDTH = int(10 * SCALING)
BALL_HEIGHT = int(10 * SCALING)

BALL_COLOR = WHITE
BALL_SHAPE = 'square'

# speed settings of the paddle and ball
PADDLE_SPEED = 1.5
BALL_X_SPEED = 3
BALL_Y_SPEED = 2
BALL_SPEED = np.sqrt(BALL_X_SPEED**2.0 + BALL_Y_SPEED**2.0)
GAMMA = 0.8
EPSILON = 1.5

BACKGROUND_COLOR = BLACK

# parameter set for the whole game
params_set = params.ParameterSet({}, label="pong_test")

BALLS_NUM = 1
PADDLES_NUM = 2

params_set['balls'] = params.ParameterSet({})
params_set['paddles'] = params.ParameterSet({})

# number of balls used in a game
params_set['balls']['num'] = BALLS_NUM

balls_list = []
params_set['balls']['list'] = balls_list

for i in range(BALLS_NUM):
    ball_params = params.ParameterSet({})
    balls_list.append(ball_params)

    ball_params['width'] = BALL_WIDTH
    ball_params['height'] = BALL_HEIGHT
    ball_params['shape'] = BALL_SHAPE
    ball_params['color'] = BALL_COLOR
    ball_params['vx'] = BALL_X_SPEED
    ball_params['vy'] = BALL_Y_SPEED
    ball_params['lbl'] = 'ball_' + str(i)


# number of paddles used in a game
params_set['paddles']['num'] = PADDLES_NUM

paddles_list = []
params_set['paddles']['list'] = paddles_list

# params for the Paddle for learning actor
paddle_params = params.ParameterSet({})
paddles_list.append(paddle_params)

paddle_params['width'] = PADDLE_WIDTH
paddle_params['height'] = PADDLE_HEIGHT
paddle_params['buffer'] = PADDLE_BUFFER
paddle_params['shape'] = PADDLE_SHAPE
paddle_params['color'] = PADDLE_COLOR
paddle_params['vx'] = 0
paddle_params['vy'] = PADDLE_SPEED
paddle_params['lbl'] = 'paddle_learn'
# convention here is that the learning paddle is on the left
paddle_params['pos'] = 'left'

# params for the Paddle for bot
paddle_params = params.ParameterSet({})
paddles_list.append(paddle_params)

paddle_params['width'] = PADDLE_WIDTH
paddle_params['height'] = PADDLE_HEIGHT
paddle_params['buffer'] = PADDLE_BUFFER
paddle_params['shape'] = PADDLE_SHAPE
paddle_params['color'] = PADDLE_COLOR
paddle_params['vx'] = 0
paddle_params['vy'] = PADDLE_SPEED
paddle_params['lbl'] = 'paddle_bot'
# convention here is that the bot opponent paddle is on the right
paddle_params['pos'] = 'right'


class Entity(object):
    def __init__(self, env, param_set):
        super(Entity, self).__init__()
        self._env = env


class Ball(Entity):
    def __init__(self, env, param_set, scale_factor):
        # env is the object corresponding to the created Pong game environment
        super(Ball, self).__init__(env, param_set)
        # self._env = env

        self.width = int(scale_factor * param_set.width)
        self.height = int(scale_factor * param_set.height)
        self.shape = param_set.shape
        self.vx = param_set.vx
        self.vy = param_set.vy
        self.color = param_set.color
        self.label = param_set.lbl
        self.vmax = scale_factor * BALL_SPEED
        self.vmin = scale_factor * (BALL_SPEED / 1.5)

        self.x = 0
        self.y = 0

        self.x_direction = self.vmin
        self.y_direction = self.vmin
        
        # initialize state for start
        self.init_state()

    def init_state(self):
        #
        WINDOW_WIDTH = self._env.SCREEN_WIDTH
        WINDOW_HEIGHT = self._env.SCREEN_HEIGHT

        BALL_WIDTH = self.width
        BALL_HEIGHT = self.height

        # random number for randomizing initial direction of ball
        # num = random.randint(0, 9)
        num = random.randint(0, 3)

        if (num == 0):
            self.x_direction = self.vmin  # self.vx stays the same ...
            self.y_direction = self.vmin  # self.vy stays the same ...
        if (num == 1):
            self.x_direction = -self.vmin  # self.vx *= -1
            self.y_direction = self.vmin
        if (num == 2):
            self.x_direction = self.vmin
            self.y_direction = -self.vmin
        if (num == 3):
            self.x_direction = -self.vmin
            self.y_direction = -self.vmin

        # new random number for randomizing vertical ball position
        num = random.randint(0, 9)
        # where the ball will start, vertical y coordinate
        self.y = num * (WINDOW_HEIGHT - BALL_HEIGHT) // 9

        # initialize the x position in the middle of the screen
        self.x = WINDOW_WIDTH // 2 - BALL_WIDTH // 2

    # reset after missing a ball if considered necessary
    def reset_after_miss(self):
        pass

    def render(self):
        pygame.draw.rect(self._env.screen, self.color, self.rect)

    def update(self):
        self.x = self.x + self.x_direction
        self.y = self.y + self.y_direction
        
    def set_vmin(self, vmin):
        self.vmin = vmin
        
    def set_vmax(self, vmax):
        self.vmax = vmax
    
    def get_angle(self):
        return np.degrees(np.arccos(np.abs(self.x_direction) / np.sqrt(self.x_direction**2.0 + self.y_direction**2.0)))

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def get_ball_velocity(self):
        return np.sqrt(self.x_direction**2.0 + self.y_direction**2.0)
    
    def get_x_direction(self):
        return self.x_direction
    
    def get_y_direction(self):
        return self.y_direction
    
    def scale_ball_velocity(self, ball_scale):
        self.vmax *= ball_scale
        self.vmin *= ball_scale
        
    def get_vmin(self):
        return self.vmin
    
    def set_color(self, color):
        self.color = color
        
class Paddle(Entity):
    def __init__(self, env, param_set, scale_factor):
        super(Paddle, self).__init__(env, param_set)

        # env is the object corresponding to the created Pong game environment
        self._env = env
        self.scale_factor = scale_factor
        self.width = int(scale_factor * param_set.width)
        self.height = int(scale_factor * param_set.height)
        self.buffer = scale_factor * param_set.buffer
        self.shape = param_set.shape
        self.vy = scale_factor * param_set.vy
        self.vmax = scale_factor * param_set.vy
        self.color = param_set.color
        self.label = param_set.lbl
        self.pos = param_set.pos  # 'left' or 'right'

        # this has to be properly initialized from the environment
        self.y = 0

        # initialize state for start
        self.init_state()

    # init state for the game start
    def init_state(self):

        # -- Set x position coordinate depending on whether paddle is right or
        # left --
        if self.pos == 'left':
            self.x = self.buffer

        if self.pos == 'right':
            self.x = self._env.SCREEN_WIDTH - self.buffer - self.width

        self.y = self._env.SCREEN_HEIGHT // 2 - self.height // 2

    def scale_height(self, scale_factor):
        self.height *= scale_factor

    # here updating paddle's own state; eventually also detecting related
    # rewards (positive score) and punishments (negative score)
    def updateAgent(self, actions, vel):
        # check whether action code is correct
        if sum(actions) != 1:
            raise ValueError('ERROR: Invalid input actions for paddle.')
        # if move up
        if (actions[1] == 1):
            self.vy = -np.abs(vel)
            self.y += self.vy
        # if move down
        if (actions[2] == 1):
            self.vy = np.abs(vel)
            self.y += self.vy

    # here updating paddle's own state; eventually also detecting related
    # rewards (positive score) and punishments (negative score)
    def updateCPU(self, actions, ball_y):
        # here eventually different controller
        # coordinates update via action IDs; action IDs provided by controller?
        # actions : e.g, [0, 1, 0]; 1 - action is selected, 0 otherwise
        # actions IDs : 0 - no op, 1 - move up; 2 - move down

        # check whether action code is correct
        if sum(actions) != 1:
            raise ValueError('ERROR: Invalid input actions for paddle.')

        # if move up
        if (actions[1] == 1):
            self.vy = np.maximum(-np.abs(ball_y), -self.vmax)
            self.y += self.vy
        # if move down
        if (actions[2] == 1):
            self.vy = np.minimum(np.abs(ball_y), self.vmax)
            self.y += self.vy

    def render(self):
        # draw paddle
        # rect gives back a pygame.Rect object through property decorator
        pygame.draw.rect(self._env.screen, self.color, self.rect)

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def set_vy(self, vy):
        self.vy = vy
        
    def get_vy(self):
        return self.vy
    
    def set_vmax(self, vmax):
        self.vmax = vmax

    def scale_vy(self, scale_factor):
        self.vy *= scale_factor

    def scale_vmax(self, scale_factor):
        self.vmax *= scale_factor

# TODO: class EnvPong(Env) - inherit from an abstract environment class
class EnvPong(object):
    """
    Class for a Simple Pong Environment

    enables switching between two different game settings
    A: classing pong. Hit and let the opponent miss to score
    B: avoidance pong. Avoid, punishment if colliding. If opponent collides with the ball, reward may be provided 		   in addition


    """

    def __init__(self, render_screen = False, screen_scale = None):
        GAMES_TO_PLAY = 1000
        GAME_LIMIT_SCORE = 5
        
        self.screen_scale = screen_scale
        (self.SCREEN_WIDTH, self.SCREEN_HEIGHT) = (int(self.screen_scale * WINDOW_WIDTH), int(self.screen_scale * WINDOW_HEIGHT))
        self.dummy = 0
        # how many games to play, until which score is one single game going
        # a constant set from properties
        self.game_rules = (GAMES_TO_PLAY, GAME_LIMIT_SCORE)

        # class Task should handle the task related stuff ?
        self.taskActive = [0, 0]  # which task is currently active
        self.taskName = ['classic', 'avoid']  # task label

        # limits of games to play and game's cycle end score for each task
        self.task_rules = [self.game_rules, self.game_rules]

        # tick counter for game duration spent in each task
        self.game_counter = [0, 0]

        # games won so far for both paddles for each task
        self.games = [[0, 0], [0, 0]]
        # current score for both paddles for each task
        self.score = [[0, 0], [0, 0]]
        
        self.ball_hits = [0, 0]
        self.ball_crossing_sides = [0, 0]
        self.ball_misses = [0, 0]
        self.ball_in_left_coll_region = False
        self.ball_in_right_coll_region = False
        self.MAX_CROSSING_LIMIT = 250
        self.agent_ball_angles = []
        self.cpu_ball_angles = []
        self.agent_ball_dir_and_vel = []
        
        self.num_balls = BALLS_NUM
        self.num_paddles = PADDLES_NUM
        self.label = 'Test_Pong_Environment_1'

        self.timer = pygame.time.Clock()
        self.render_screen = render_screen
        
        self.balls = []

        for i in range(self.num_balls):
            ball = Ball(self, params_set.balls.list[i], self.screen_scale)
            self.balls.append(ball)

        self.paddles = []

        for i in range(self.num_paddles):
            paddle = Paddle(self, params_set.paddles.list[i], self.screen_scale)
            self.paddles.append(paddle)

        self.screen = None
        self.display = None
                
        # initialize pygame screen and display
        if self.render_screen:
            self.initialize_display_for_rendering()
        else:
            self.initializeDisplay()
        self.player_one_score = 0
        self.player_two_score = 0
        
        self.paddles[0].set_vy(screen_scale * 5)
        
        self.epsilon = screen_scale * EPSILON
       
    def get_ball_hits(self):
        return self.ball_hits

    def scale_ball_velocity(self, scale_factor):
        ball = self.get_ball()
        ball.scale_ball_velocity(scale_factor)
        self.epsilon *= scale_factor

    def get_ball_misses(self):
        return self.ball_misses
    
    def get_ball_crossing_sides(self):
        return self.ball_crossing_sides
    
    def get_agent_ball_angles(self):
        return self.agent_ball_angles
    
    def get_cpu_ball_angles(self):
        return self.cpu_ball_angles
    
    def get_agent_ball_dir_and_vel(self):
        return self.agent_ball_dir_and_vel
    
    def get_ball(self, i=0):
        return self.balls[i]
    
    def setFloat(self, rap_utf8, rap):
        pass
    
    def setInt(self, b_random_seed, seed):
        pass
    
    def loadROM(self, game_path):
        pass
    
    def getMinimalActionSet(self):
        return [[0, 1, 0], [0, 0, 1]]
    
    def getScreenDims(self):
        return (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
    
    def set_ball_color(self, color):
        ball = self.balls[0]
        ball.set_color(color)

    def scale_paddles(self, scale_factor):
        for paddle in self.paddles:
            paddle.scale_height(scale_factor)

    def scale_paddle_vel(self, scale_factor):
        for paddle in self.paddles:
            paddle.scale_vy(scale_factor)
            paddle.scale_vmax(scale_factor)

    def reset_game(self):
        ball = self.balls[0]
        speed = ball.get_vmin()
        ball.x_direction = -random.uniform(self.epsilon, np.sqrt(speed**2.0 - self.epsilon**2.0))
        ball.y_direction = random.choice([-1, 1]) * np.sqrt(speed**2.0 - ball.x_direction**2.0)
        ball.x = self.SCREEN_WIDTH // 2 - ball.width // 2
        ball.y = random.randrange(0, self.SCREEN_HEIGHT - ball.height)
        
        paddle_learner = self.paddles[0]
        paddle_bot = self.paddles[1]

        paddle_learner.y = ball.y - paddle_learner.height // 2
        paddle_bot.y = ball.y - paddle_bot.height // 2
        
        self.player_one_score = 0
        self.player_two_score = 0
        
        self.ball_hits = [0, 0]
        self.ball_crossing_sides = [0, 0]
        self.ball_misses = [0, 0]
        self.ball_in_left_coll_region = False
        self.ball_in_right_coll_region = False
        self.agent_ball_angles = []
        self.cpu_ball_angles = []
        self.agent_ball_dir_and_vel = []
        
    def getScreenRGB2(self):
        return self.render()
    
    def act(self, action):
        actions = [action, [1, 0, 0]]
            
        paddle_bot = self.paddles[1]    
        player_bot_controller = EnvPongDraft.PaddleBot_Controller(self, paddle_bot, params.ParameterSet({}))    
        actions[1] = player_bot_controller.act()
        
        self.updateEntities(actions)
        score_reward, hit_reward = self.detect_events()
        #reward = score_reward + 0.5 * hit_reward
        reward = score_reward
        
        return reward
    
    def reset_ball(self, signum):
        ball = self.balls[0]
        speed = ball.get_vmin()
        ball.x_direction = signum * random.uniform(self.epsilon, np.sqrt(speed**2.0 - self.epsilon**2.0))
        ball.y_direction = random.choice([-1, 1]) * np.sqrt(speed**2.0 - ball.x_direction**2.0)
        ball.x = self.SCREEN_WIDTH // 2 - ball.width // 2
        ball.y = random.randrange(0, self.SCREEN_HEIGHT - ball.height)
        if signum > 0.0:
            self.ball_crossing_sides[1] += 1
        else:
            self.ball_crossing_sides[0] += 1
    
    def game_over(self):
        return self.player_one_score >= 20 or self.player_two_score >= 20 or np.sum(self.ball_crossing_sides) >= self.MAX_CROSSING_LIMIT
    
    def lives(self):
        return None
        
    # draw the entities / objects like ball, paddles on the screen
    def renderEntities(self):
        # draw balls
        for i in range(self.num_balls):
            self.balls[i].render()

        # draw paddles
        for i in range(self.num_paddles):
            self.paddles[i].render()

    # update the state of the entities / objects like ball, paddles
    def updateEntities(self, actions):
        # update balls
        for i in range(self.num_balls):
            self.balls[i].update()

        # check whether action code is correct
        if len(actions) != self.num_paddles:
            raise ValueError('ERROR: number of action set does not match number of paddles.')

        # Update paddles
        # Update Agent
        self.paddles[0].updateAgent(actions[0], self.paddles[0].get_vy())

        # Update CPU
        self.paddles[1].updateCPU(actions[1], self.balls[0].y_direction)

    def initializeDisplay(self):
        # first or both will make it headless, but then no output to surface possible
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # os.putenv('SDL_VIDEODRIVER', 'fbcon')

        # this would create a window
        # os.putenv('DISPLAY', ':0.0')

        # initialize the screen using width and height variables
        # TEST: if disabling it, game still can write to surface?
        # self.display = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        # pygame.display.init()
        # this has to be done to initialize a dummy display
        self.display = pygame.display.set_mode((1, 1))

        # prepare the surface for game images
        # this does not work
        # self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT),  pygame.SRCALPHA, 32)
        # adding pygame.SRCALPHA, 32 fixes it. So surface needs a canvas to paint on without display
        # depth 16 works, too
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA, 32)
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0)

        # for each frame, calls the event queue, like if the main window needs to be repainted
        pygame.event.pump()
        # make the background black
        self.screen.fill(BLACK)

        # draw entities (balls and paddles)
        self.renderEntities()

        # updates the window
        # pygame.display.flip()
        
    def initialize_display_for_rendering(self):
        # initialize the screen using width and height variables
        flags = DOUBLEBUF
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), flags)

        # for each frame, calls the event queue, like if the main window needs to be repainted
        pygame.event.pump()
        # make the background black
        self.screen.fill(BLACK)

        # draw entities (balls and paddles)
        self.renderEntities()

        # updates the window
        pygame.display.flip()

    # for saving an image to disk
    def save_image(self, filename):
        pygame.image.save(self.screen, filename)

    def render(self, render_screen = False):
        # for each frame, calls the event queue, like if the main window needs to be repainted
        pygame.event.pump()
        # make the background black
        self.screen.fill(BLACK)
        # render the entities (balls, paddles)
        self.renderEntities()
        # copy the pixels from game surface to a 3D array. Can be used as input
        # for the network
        # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        image_data = pygame.surfarray.array3d(self.screen)

        # updates the window
        if self.render_screen:
            pygame.display.flip()
        # return our surface data
        return image_data
    
    def reset(self):
        pass

    # handle collisions, provide reward and punishment
    def detect_events(self):
        # returns reward ; [pos, neg]
        # positive (ID 0) and negative (ID 1) reward for each paddle
        # paddle ID 0 : learning; paddle ID 1: bot
        # reward = [0, 0]
        reward = [[0, 0], [0, 0]]
        # TODO: here collecting ball hit events for learner and bot paddles
        hits = [0, 0]

        ball = self.balls[0]
        paddle_learner = self.paddles[0]
        paddle_bot = self.paddles[1]

        ballXPos = ball.x
        ballYPos = ball.y

        ballXDirection = ball.x_direction
        ballYDirection = ball.y_direction

        BALL_HEIGHT = ball.height
        BALL_WIDTH = ball.width

        PADDLE_BUFFER = paddle_learner.buffer
        PADDLE_WIDTH = paddle_learner.width
        PADDLE_HEIGHT = paddle_learner.height

        WINDOW_WIDTH = self.SCREEN_WIDTH
        WINDOW_HEIGHT = self.SCREEN_HEIGHT

        # learner paddle, by convention on the left
        paddle_learner_XPos = paddle_learner.x
        paddle_learner_YPos = paddle_learner.y

        # bot paddle, by convention on the right
        paddle_bot_XPos = paddle_bot.x
        paddle_bot_YPos = paddle_bot.y
        hit_reward = 0.0
        score_reward = 0.0
        # learner_score = 0
        # opponent_score = 0
        # paddle_score = [0, 0] # score for learner and opponent
        # paddle : learning agent (or human control) and bot opponent (or human control)
        
        
        if ball.x_direction > 0.0 and ball.x > (self.SCREEN_WIDTH // 2 - ball.width // 2):
            self.ball_in_left_coll_region = False
            
        if ball.x_direction < 0.0 and ball.x < (self.SCREEN_WIDTH // 2 - ball.width // 2):
            self.ball_in_right_coll_region = False
            
        if ball.x_direction > 0.0 and ball.x > (self.SCREEN_WIDTH // 4 + 0.5 * paddle_learner_XPos) and ball.x < (self.SCREEN_WIDTH // 2 - ball.width // 2) and not self.ball_in_left_coll_region:
            self.ball_in_left_coll_region = True
            self.ball_crossing_sides[0] += 1
            #print(self.ball_crossing_sides[0])
            
        if ball.x_direction < 0.0 and ball.x < (self.SCREEN_WIDTH // 4 + 0.5 * paddle_bot_XPos) and ball.x > (self.SCREEN_WIDTH // 2 - ball.width // 2) and not self.ball_in_right_coll_region:
            self.ball_in_right_coll_region = True
            self.ball_crossing_sides[1] += 1
            #print(self.ball_crossing_sides[1])
        

        # checks for a collision, if the ball hits the Player side - the learning agent        
        if (ballXPos <= paddle_learner_XPos + PADDLE_WIDTH and ballYPos + BALL_HEIGHT >= paddle_learner_YPos and ballYPos - BALL_HEIGHT <= paddle_learner_YPos + PADDLE_HEIGHT and ballXDirection < 0.0):
            # case : ball caught by the learning agent paddle (by convention: on the left side)
            # reward situation for classic for case of ball catching:
            # - either no reward (harder case, no intermediate information)
            # - or small reward to indicate that catching ball is a good thing toward winning
            # (main principle : keep reward sensation local to the agent ?)


            # when considering both (in this case, when opponent misses, score_reward should be higher than hit_reward)
            score_reward = 0

            # when ball rewards only
            hit_reward = 1.0

            # switches directions (ATT: make sure this has effect on the original object)
            ball.x_direction *= -1.0
            ball.y_direction += GAMMA * paddle_learner.vy
            norm = np.sqrt(ball.x_direction ** 2.0 + ball.y_direction ** 2.0)
            if norm > ball.vmax:
                ball.x_direction = (ball.x_direction / norm) * ball.vmax
                ball.y_direction = (ball.y_direction / norm) * ball.vmax
                norm = ball.vmax
            elif norm < ball.vmin:
                ball.x_direction = (ball.x_direction / norm) * ball.vmin
                ball.y_direction = (ball.y_direction / norm) * ball.vmin
                norm = ball.vmin
            if np.abs(ball.x_direction) < self.epsilon:
                ball.x_direction = np.sign(ball.x_direction) * random.uniform(self.epsilon, np.sqrt(
                    norm ** 2.0 - self.epsilon ** 2.0))
                ball.y_direction = np.sign(ball.y_direction) * np.sqrt(norm ** 2.0 - ball.x_direction ** 2.0)

            # hit counter for learner paddle
            hits[0] += 1
            self.ball_hits[0] += 1
            self.agent_ball_angles.append(ball.get_angle())
            self.agent_ball_dir_and_vel.append([ball.x_direction, ball.y_direction, ball.get_ball_velocity()])
            # ---- here depending on the task ---------------
            # if task is to avoid, negative score for the learning agent
            # self.score[0] -= 1

            # --- learner catching the ball ---
            if self.taskActive[0]:
                # -- classical pong
                self.total_hits[0][0] += 1
                # eventualy, help positive reward for learner catching the ball
                # reward[0][0] += 1
            elif self.taskActive[1]:
                # -- avoidance pong
                self.total_hits[1][0] += 1
                # avoidance pong, learner collided with the ball
                # learner could not avoid the ball, punish the learner?
                # reward for the opponent
                reward[0][1] += -1
                reward[1][0] += 1
                self.total_reward[1][0][1] += -1
                self.total_reward[1][1][0] += 1


        # if player misses/avoids the ball
        elif (ballXPos <= 0):
            # case : ball missed by the learning agent paddle (by convention: on the left side)
            # reward situation for classic for case of missing the ball:
            # - punishment, negative reward - loosing situation (main principle : keep reward sensation local to the agent ?)
            # (although the agent anyway perceives the whole field; locality would make sense if one would require the agent's body is the paddle and every sensory input should be treated as coming through the paddle's "body")

            score_reward = -1.0
            self.player_two_score += 1
            self.reset_ball(-1.0)
            
            # when ball rewards only
            hit_reward = -1.0
            
            self.ball_misses[0] += 1

            # ---- here depending on the task ---------------
            # if task is to hit, negative score for the learning agent
            # if the task is to avoid, everything is okay, neutral

            # self.score[0] -= 1
            # learner_score = -1.0  # for image based state learning
            # negative reward for learner agent for missing the ball

            # --- learner missing the ball ---
            if self.taskActive[0]:
                # learner missed the ball : classic pong
                # negative reward for the learner
                # reward IDs: [paddle_ID][reward_type_ID]
                # paddle_ID : 0 (learner), 1 (opponent)
                # reward_type_ID : 0 (reward), 1 (punishment)
                reward[0][1] += -1
                # positive reward for the opponent
                reward[1][0] += 1
                self.total_reward[0][0][1] += -1
                self.total_reward[0][1][0] += 1
            elif self.taskActive[1]:
                # learner missed the ball : avoidance pong
                pass

        # check if ball hits the opponent player
        if (
            ballXPos >= paddle_bot_XPos and ballYPos + BALL_HEIGHT >= paddle_bot_YPos and ballYPos - BALL_HEIGHT <= paddle_bot_YPos + PADDLE_HEIGHT and ballXDirection >= 0.0):
            # case : ball caught by the ** opponent ** agent paddle (here by convention: on the right side)
            # reward situation for classic for case of ball catching by opponent:
            # - no reward for the learning agent (a smaller reward should be provided to opponent in case it learns, too)
            # - an interesting intermediate modification may be a negative reward for the learning agent, which should push forward strategies to hit the ball back in the way to make it hard for the opponent to catch it

            # switch directions
            ball.x_direction *= -1.0
            ball.y_direction += GAMMA * paddle_bot.vy
            norm = np.sqrt(ball.x_direction ** 2.0 + ball.y_direction ** 2.0)
            if norm > ball.vmax:
                ball.x_direction = (ball.x_direction / norm) * ball.vmax
                ball.y_direction = (ball.y_direction / norm) * ball.vmax
                norm = ball.vmax
            elif norm < ball.vmin:
                ball.x_direction = (ball.x_direction / norm) * ball.vmin
                ball.y_direction = (ball.y_direction / norm) * ball.vmin
                norm = ball.vmin
            if np.abs(ball.x_direction) < self.epsilon:
                ball.x_direction = np.sign(ball.x_direction) * random.uniform(self.epsilon, np.sqrt(
                    norm ** 2.0 - self.epsilon**2.0))
                ball.y_direction = np.sign(ball.y_direction) * np.sqrt(norm ** 2.0 - ball.x_direction ** 2.0)
            # can be outcommented if no visual debugging - signaling a loss by ball color -  is needed
            # NewBallColor = WHITE

            # hit counter for opponent paddle
            hits[1] += 1
            self.cpu_ball_angles.append(ball.get_angle())
            # ---- here depending on the task ---------------
            # in avoid task, additional reward maybe provided if opponent collided with the ball
            # reward[0] = 1
            # if the task is to avoid, opponent gets punished, score down :
            # self.game_score[1] -= 1
            # or alternatively, score up for the learner : (game is over
            # once a player was hit certain number of times HITS, SCORE_LIMIT)
            # reward[0] = 1
            # self.game_score[0] += 1
            # --- opponent catching the ball
            if self.taskActive[0]:
                # total_hits IDs : [task_ID][paddle_ID]
                self.total_hits[0][1] += 1
            elif self.taskActive[1]:
                self.total_hits[1][1] += 1
                # opponent could not avoid the ball, punish the opponent?
                # positive reward for the learner? yes, for the symmetry
                reward[0][0] += 1
                reward[1][1] += -1
                self.total_reward[1][0][0] += 1
                self.total_reward[1][1][1] += -1
            self.ball_hits[1] += 1


        # opponent missing the ball
        elif (ballXPos >= WINDOW_WIDTH - BALL_WIDTH):
            # positive reward for the learning agent (actually, against the local reward processing principle, as ball miss is happening at the opponent)

            # when taking into account score reward:
            score_reward = 1.0
            self.player_one_score += 1
            self.reset_ball(1.0)

            # when ball rewards only
            hit_reward = 0
            
            self.ball_misses[1] += 1

            # can be outcommented if no visual debugging - signaling a loss by ball color -  is needed
            # NewBallColor = WHITE

            # ----- after missing a ball : can be considered as end of game episode,
            # - either reset the game to random or specified initial conditions
            # - or continue using the ball just bumping off from the screen edge
            # any reason one should prefer reset over non-reset or other way around?

            # ---- here depending on the task ---------------
            # in classical pong, opponent missing a ball gives positive reward
            # and score to the learning agent

            # opponent missing the ball :
            if self.taskActive[0]:
                # opponent missed the ball : classic pong
                # positive reward for the learner
                reward[0][0] += 1
                # negative reward for the opponent
                reward[1][1] += -1
                self.total_reward[0][0][0] += 1
                self.total_reward[0][1][1] += -1
            elif self.taskActive[1]:
                # opponent missed the ball : avoidance pong
                pass

        # if the ball hits the top screen edge - move down
        if (ballYPos <= 0):
            ball.y = 0
            ball.y_direction *= -1.0  # downwards direction from top to bottom is positive, 1
            # return reward
        # if the ball hits the bottom screen edge -  move up
        elif (ballYPos >= WINDOW_HEIGHT - BALL_HEIGHT):
            ball.y = WINDOW_HEIGHT - BALL_HEIGHT
            # upwards direction from bottom to top is negative, -1
            ball.y_direction *= -1.0

        # *** --------- now paddle collision detection ---------- ***
        # moving the code here because eventually rewards and punishments maybe
        # also provided here
        # don't let the learning paddle (on the left) move off the screen
        # don't let it hit the top
        if (paddle_learner_YPos < 0):
            paddle_learner.y = 0
        # don't let it hit the bottom
        if (paddle_learner_YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
            paddle_learner.y = WINDOW_HEIGHT - PADDLE_HEIGHT

        # don't let bot paddle hit the top
        if (paddle_bot_YPos < 0):
            paddle_bot.y = 0
        # dont let it hit the bottom
        if (paddle_bot_YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
            paddle_bot.y = WINDOW_HEIGHT - PADDLE_HEIGHT
        
        
        
        return score_reward, hit_reward