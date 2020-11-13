#
# Simple Pong Game Environment Draft based upon PyGame
# Environment Pong Game
#

# author: Jenia Jitsev, 2017, 2018

from __future__ import absolute_import, division, print_function, unicode_literals
# This ensures Python 2, 3 compatible code

import pygame
import random

import numpy as np
import parameters as params

from pygame.locals import *



# try:
# 	param_filename = str(sys.argv[ARG_PARAM_FILE_POS])
# 	paramset = __import__(param_filename)
# except IndexError:
# 	print 'ERROR : wrong argument index addressed.'
# 	sys.exit('Usage: %s PARAM_FILE [optional: SEED_NUM PROC_NUM]' %sys.argv[0])
#
# except ImportError:
# 	print 'ERROR : paramater file provided is invalid.'
# 	print '(file name was : ', param_filename, ' ).'
# 	sys.exit('Usage: %s PARAM_FILE [optional: SEED_NUM PROC_NUM]' %sys.argv[0])

# frames rate per second
FPS = 60

# size of the game screen (in case it is square)
SCALING = 1.0
SCALING = 0.2
SCREEN_SIZE = int(400 * SCALING)

# size of the game screen
WINDOW_WIDTH = SCREEN_SIZE + 23
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
BALL_WIDTH = int(10 *SCALING)
BALL_HEIGHT = int(10 * SCALING)

BALL_COLOR = WHITE
BALL_SHAPE = 'square'

# speed settings of the paddle and ball
PADDLE_SPEED = 2
BALL_X_SPEED = 3
BALL_Y_SPEED = 2
BALL_SPEED = np.sqrt(BALL_X_SPEED**2.0 + BALL_Y_SPEED**2.0)
GAMMA = 0.1
EPSILON = 1.0

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
    # cannot call it label, as the name is already reserved
    ball_params['lbl'] = 'ball_' + str(i)

# testing params
# print 'ball param_set label (in the very beginning): ', params_set.balls.list[0].label
# print 'ball param_set label (in the very beginning): ', params_set.balls.list[0].lbl
# print 'ball param_set BALL SHAPE (in the very beginning): ', params_set.balls.list[0].shape

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


class Controller(Entity):
    def __init__(self, env, ent, param_set):
        # env is the object corresponding to the created Pong game environment
        super(Controller, self).__init__(env, param_set)

        # Entity that is controlled by the respective Controller
        self._ent = ent

        # eventually, setting also the Entity's internal variable to the
        # controler
        # ent._controller = self
        # TODO: introduce a controler in Entity ?
        # Or in Controllable_Entities only

    def act(self):
        """
        Performs actions given state of the environment and other variables
        Actions are returned
        """
        abstract

class PaddleBot_Controller(Controller):
    def __init__(self, env, ent, param_set):
        # env is the object corresponding to the created Pong game environment
        super(PaddleBot_Controller, self).__init__(env, ent, param_set)

        # Entity that is controlled by the respective Controller

    def act(self):
        # hard wired catch-the-ball strategy

        env = self._env
        ball = env.balls[0]
        paddle = self._ent

        BALL_HEIGHT = ball.height
        BALL_WIDTH = ball.width

        PADDLE_BUFFER = paddle.buffer
        PADDLE_WIDTH = paddle.width
        PADDLE_HEIGHT = paddle.height

        WINDOW_WIDTH = env.SCREEN_WIDTH
        WINDOW_HEIGHT = env.SCREEN_HEIGHT

        ballXPos = ball.x
        ballYPos = ball.y

        # actions IDs : 0 - no op, 1 - move up; 2 - move down
        # default : do nothing (ID : 0)
        actions = [1, 0, 0]

        # simple strategy to catch the ball
        # move down if ball is below the paddle
        if (paddle.y + PADDLE_HEIGHT // 2 < ballYPos + BALL_HEIGHT // 2):
            # paddle2YPos = paddle2YPos + PADDLE_SPEED
            actions = [0, 0, 1]
        # move up if ball is above the paddle
        if (paddle.y + PADDLE_HEIGHT // 2 > ballYPos + BALL_HEIGHT // 2):
            # paddle2YPos = paddle2YPos - PADDLE_SPEED
            actions = [0, 1, 0]

        # ball is below the paddle
        # if (paddle.y + PADDLE_HEIGHT // 2 < ballYPos + BALL_HEIGHT // 2):
        #     # ball flying down
        #     if ball.y_direction == 1 :
        #         # move up ?
        #         actions = [0, 1, 0]
        #     # ball flying up
        #     elif ball.y_direction == -1 :
        #         # move down ?
        #         actions = [0, 0, 1]
        #
        #     # paddle2YPos = paddle2YPos + PADDLE_SPEED
        #     # actions = [0, 0, 1]
        # # move up if ball is above the paddle
        # if (paddle.y + PADDLE_HEIGHT // 2 > ballYPos + BALL_HEIGHT // 2):
        #     # paddle2YPos = paddle2YPos - PADDLE_SPEED
        #     # actions = [0, 1, 0]
        #    # ball flying down
        #    if ball.y_direction == 1 :
        #        # move down
        #        actions = [0, 0, 1]
        #    # ball flying up
        #    elif ball.y_direction == -1 :
        #        # move up
        #        actions = [0, 1, 0]

        return actions

class PaddleBotRandom_Controller(Controller):
    def __init__(self, env, ent, param_set):
        # env is the object corresponding to the created Pong game environment
        super(PaddleBotRandom_Controller, self).__init__(env, ent, param_set)

        # Entity that is controlled by the respective Controller

    def act(self):
        # choose a random action for paddle

        actions = [0, 0, 0]

        ind_act = random.randint(0, 2)

        actions[ind_act] = 1

        # actions IDs : 0 - no op, 1 - move up; 2 - move down
        # default : do nothing (ID : 0)


        return actions

# this is a controller for avoiding the ball, as opposed to classical case (
# where the ball is catched)
#
class PaddleBotAvoid_Controller(Controller):
    def __init__(self, env, ent, param_set):
        # env is the object corresponding to the created Pong game environment
        super(PaddleBotAvoid_Controller, self).__init__(env, ent, param_set)

        # Entity that is controlled by the respective Controller

    def act(self):
        # hard wired catch-the-ball strategy

        env = self._env
        ball = env.balls[0]
        paddle = self._ent

        BALL_HEIGHT = ball.height
        BALL_WIDTH = ball.width

        PADDLE_BUFFER = paddle.buffer
        PADDLE_WIDTH = paddle.width
        PADDLE_HEIGHT = paddle.height

        WINDOW_WIDTH = env.SCREEN_WIDTH
        WINDOW_HEIGHT = env.SCREEN_HEIGHT

        ballXPos = ball.x
        ballYPos = ball.y

        # actions IDs : 0 - no op, 1 - move up; 2 - move down
        # default : do nothing (ID : 0)
        actions = [1, 0, 0]

        # simple strategy to avoid collision with the ball
        # move up if ball is below the paddle

        # strategy based only on ball direction (move away from where ball goes)
        # ball flying down
        if ball.y_direction >= 0.0 :
            # move up
            actions = [0, 1, 0]
        # ball flying up
        elif ball.y_direction < 0.0 :
            # move down
            actions = [0, 0, 1]

        # strategy based on relative position of paddle and ball (works slightly
        # less well for avoidance)
        #
        # move up if ball is below the paddle
        # if (paddle.y + PADDLE_HEIGHT // 2 < ballYPos + BALL_HEIGHT // 2):
        #     # paddle2YPos = paddle2YPos + PADDLE_SPEED
        #     actions = [0, 1, 0]
        # # move down if ball is above the paddle
        # if (paddle.y + PADDLE_HEIGHT // 2 > ballYPos + BALL_HEIGHT // 2):
        #     # paddle2YPos = paddle2YPos - PADDLE_SPEED
        #     actions = [0, 0, 1]

        return actions


class Entity_Updatable(Entity):
    def __init__(self, env, param_set):
        super(Entity_Updatable, self).__init__(env, param_set)

    def update(self, actions):
        abstract



class Ball(Entity_Updatable):
    def __init__(self, env, param_set):
        # env is the object corresponding to the created Pong game environment
        super(Ball, self).__init__(env, param_set)
        # self._env = env

        self.width = param_set.width
        self.height = param_set.height
        self.shape = param_set.shape
        self.vx = param_set.vx
        self.vy = param_set.vy
        self.color = param_set.color
        self.label = param_set.lbl

        # print 'Within Ball Class Init, param_set label: ', param_set.label

        # testing properties
        # print 'Within Ball Class Init, Color: ', self.color
        # print 'Within Ball Class Init, label: ', self.label

        # this has to be properly initialized from the environment
        self.x = 0
        self.y = 0

        self.x_direction = BALL_X_SPEED
        self.y_direction = BALL_Y_SPEED

        # initialize state for start
        self.init_state()
        # print 'Test INIT BALL'

    # init ball state for the game start
    def init_state(self):
        #
        WINDOW_WIDTH = self._env.SCREEN_WIDTH
        WINDOW_HEIGHT = self._env.SCREEN_HEIGHT

        BALL_WIDTH = self.width
        BALL_HEIGHT = self.height

        # random number for randomizing initial direction of ball
        # num = random.randint(0, 9)
        num = random.randint(0, 3)

        # randomly decide where the ball will move
        # it is better to introduce direction variables - for the case
        # one would like to vary speed amplitude
        # if(0 <= num < 3):
        #     self.x_direction = 1 # self.vx stays the same ...
        #     self.y_direction = 1 # self.vy stays the same ...
        # if (3 <= num < 5):
        #     self.x_direction = -1 # self.vx *= -1
        #     self.y_direction = 1
        # if (5 <= num < 8):
        #     self.x_direction = 1
        #     self.y_direction = -1
        # if (8 <= num < 10):
        #     self.x_direction = -1
        #     self.y_direction = -1

        if (num == 0):
            self.x_direction = BALL_X_SPEED # self.vx stays the same ...
            self.y_direction = BALL_Y_SPEED # self.vy stays the same ...
        if (num == 1):
            self.x_direction = -BALL_X_SPEED # self.vx *= -1
            self.y_direction = BALL_Y_SPEED
        if (num == 2):
            self.x_direction = BALL_X_SPEED
            self.y_direction = -BALL_Y_SPEED
        if (num == 3):
            self.x_direction = -BALL_X_SPEED
            self.y_direction = -BALL_Y_SPEED

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
        # draw ball
        # right now only rect form is supported
        # rect gives back a pygame.Rect object through property decorator
        pygame.draw.rect(self._env.screen, self.color, self.rect)

        # an option for ellipse (define corresponding elps property ?)
        # pygame.draw.ellipse(self._env.screen, self.color, self.rect)
        # pygame.draw.ellipse(self._env.screen, self.color, self.elps)

    # here updating balls's own state;
    # should state update also be hanled by a controller?
    # detecting possible collision events (with screen walls, paddles)
    # Also detecting related rewards (positive score) and punishments (negative score)
    # ball hit, ball miss are the most relevant events
    def update(self, actions = None):

        # dft = 7.5

        # update the x and y position
        # ballXPos = ballXPos + ballXDirection * BALL_X_SPEED * dft
        # ballYPos = ballYPos + ballYDirection * BALL_Y_SPEED * dft
        # score = 0

        # NewBallColor = BallColor

        # update the x and y position
        # ballXPos = self.x + self.x_direction * self.vx
        # ballYPos = self.y + self.y_direction * self.vy
        #self.x = self.x + self.x_direction * self.vx
        #self.y = self.y + self.y_direction * self.vy
        self.x = self.x + self.x_direction
        self.y = self.y + self.y_direction


        # maybe makes more sense to do collision detection in Pong class ?
        # so here updating coordinates only ?

        # PADDLE_BUFFER = self._env.get_paddle_buffer()

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Paddle(Entity_Updatable):
    def __init__(self, env, param_set):
        super(Paddle, self).__init__(env, param_set)

        # env is the object corresponding to the created Pong game environment
        self._env = env

        self.width = param_set.width
        self.height = param_set.height
        self.buffer = param_set.buffer
        self.shape = param_set.shape
        self.vy = param_set.vy
        self.color = param_set.color
        self.label = param_set.lbl
        self.pos = param_set.pos # 'left' or 'right'

        # this has to be properly initialized from the environment
        self.y = 0

        # initialize state for start
        self.init_state()

    # init state for the game start
    def init_state(self):

        # -- Set x position coordinate depending on whether paddle is right or
        # left --

        if self.pos == 'left':
            # on the left, x coord corresponds to shift of paddle given by its
            # distance from screen edge (buffer)
            # width is not take into account because rect will be drawn
            # extending to the right
            self.x = self.buffer

        if self.pos == 'right':
            self.x = self._env.SCREEN_WIDTH - self.buffer - self.width

        self.y = self._env.SCREEN_HEIGHT // 2 - self.height // 2

        # initialie positions of paddle
        # self.paddle1YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        # self.paddle2YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2

        # random number for initial position of paddle 2
        # num = random.randint(0, 9)
        # self.paddle2YPos = num * (WINDOW_HEIGHT - PADDLE_HEIGHT) / 9

    # here updating paddle's own state; eventually also detecting related
    # rewards (positive score) and punishments (negative score)
    def update(self, actions):
        # here eventually different controller
        # coordinates update via action IDs; action IDs provided by controller?
        # actions : e.g, [0, 1, 0]; 1 - action is selected, 0 otherwise
        # actions IDs : 0 - no op, 1 - move up; 2 - move down

        # check whether action code is correct
        if sum(actions) != 1:
            raise ValueError('ERROR: Invalid input actions for paddle.')

        # if move up
        if (actions[1] == 1):
            self.vy = -np.abs(self.vy)
            self.y += self.vy
        # if move down
        if (actions[2] == 1):
            self.vy = np.abs(self.vy)
            self.y += self.vy


    def render(self):
        # draw paddle
        # rect gives back a pygame.Rect object through property decorator
        pygame.draw.rect(self._env.screen, self.color, self.rect)

    @property
    def rect(self):
        # pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddle2YPos, PADDLE_WIDTH, PADDLE_HEIGHT)
        # WINDOW_WIDTH = self._env.SCREEN_WIDTH
        # PADDLE_WIDTH = self.width
        # PADDLE_BUFFER = self.buffer

        # so here paddle can be either on left side, or right side
        # depending on that, different fixed x coordinate has to be returned
        # better to handle in self.x init, providing here self.x only
        # return pygame.Rect(self._env.SCREEN_WIDTH - self.buffer - self.width, self.y, self.width, self.height)

        # self.x is set in the init method depending on left or right paddle property
        #
        # the only changing coordinate here is self.y
        return pygame.Rect(self.x, self.y, self.width, self.height)

# TODO: class EnvPong(Env) - inherit from an abstract environment class
class EnvPong(object):

    """
    Class for a Simple Pong Environment

    enables switching between two different game settings
    A: classing pong. Hit and let the opponent miss to score
    B: avoidance pong. Avoid, punishment if colliding. If opponent collides with the ball, reward may be provided 		   in addition


    """

    def __init__(self):
        # total games : one game is over when limit score is reached by
        # one of the players
        GAMES_TO_PLAY = 1000
        # limit score to play until for one game
        GAME_LIMIT_SCORE = 5

        self.dummy = 0
        # how many games to play, until which score is one single game going
        # a constant set from properties
        self.game_rules = (GAMES_TO_PLAY, GAME_LIMIT_SCORE)

        # class Task should handle the task related stuff ?
        # task ID 0: classic pong; ID 1: avoid ball
        self.taskActive = [True, False]  # which task is currently active
        self.taskName = ['classic', 'avoid']  # task label

        # limits of games to play and game's cycle end score for each task
        self.task_rules = [self.game_rules, self.game_rules]

        # indicates task switch state:
        # True : switch the task to the next
        # False : continue with the current task
        self.task_switch = False
        # whether instance is running in multi task mode

        # if True, task_switch condition has to be checked
        # if False, just proceed with the preselected task
        self.multi_task = False

        # counting events to measure how long the task is executed
        self.task_iter_counter = 0

        # counter for ball crossing the midline in each task
        self.ball_cross_counter = [0, 0]

        # tick counter for game duration spent in each task
        self.game_counter = [0, 0]

        # self.games = [0, 0]  # games won so far for both paddles
        # self.score = [0, 0]  # current score for both paddles

        # should one have scores also separate for each task?
        # as it will be important to see which task is running well,
        # and which is failing.
        # This is not apparent from the common score
        # A counter for total games played could be convinient
        # although total games can be inferred from the number of won games
        # for each paddle

        # games won so far for both paddles for each task
        self.games = [[0, 0],  [0, 0]]
        # current score for both paddles for each task
        self.score = [[0, 0],  [0, 0]]

        # hits on paddle in total, for each task, each paddle
        # ID 0 : classic pong, ID 1 : avoid
        # ID -, 0 : learner; ID -, 1 : opponent;
        # e.g : self.hits[0][0] : hits for learning in classic pong
        self.total_hits = [[0, 0],  [0, 0]]
        # pos, neg reward for each paddle, each task
        # e.g self.reward[0][0][0] (task, paddle, pos/neg)
        self.total_reward = [[[0, 0],  [0, 0]], [[0, 0],  [0, 0]]]

        self.num_balls = BALLS_NUM
        self.num_paddles = PADDLES_NUM
        self.label = 'Test_Pong_Environment_1'

        (self.SCREEN_WIDTH, self.SCREEN_HEIGHT) = (WINDOW_WIDTH, WINDOW_HEIGHT)

        # game timer
        self.timer = pygame.time.Clock()

        # self.screen_width = WINDOW_WIDTH

        # ball = Ball(self, params_set.balls.list[0])
        # self.screen_height = WINDOW_HEIGHT

        # fog, "devil blows" could introduce partial information and uncertainty
        # into the game

        # self.Balls = [ball]
        self.balls = []

        for i in range(self.num_balls):
            ball = Ball(self, params_set.balls.list[i])
            self.balls.append(ball)

        self.paddles = []

        for i in range(self.num_paddles):
            paddle = Paddle(self, params_set.paddles.list[i])
            self.paddles.append(paddle)

        # print 'Ball ID: ', ball.label
        # testing ball properties
        """
        for i in range(self.num_balls):
            print()
            print('Ball ID: ', self.balls[i].label)
            print('Ball width: ', self.balls[i].width)
            print('Ball height: ', self.balls[i].height)
            print('Ball shape: ', self.balls[i].shape)
            print('Ball Environment: ', self.balls[i]._env.label)
            print()

        # testing Paddle properties

        for i in range(self.num_paddles):
            print()
            print('Paddle ID: ', self.paddles[i].label)
            print('Paddle width: ', self.paddles[i].width)
            print('Paddle height: ', self.paddles[i].height)
            print('Paddle shape: ', self.paddles[i].shape)
            print('Paddle position: ', self.paddles[i].pos)
            print('Paddle Environment: ', self.paddles[i]._env.label)
            print()
        """
        # self.Ball = None
        # self.Paddle = None
        # self.Paddles = [None, None]
        # in case multiple Balles present : a list ?
        # self.Balles = []

        self.screen = None
        # initialize pygame screen and display
        self.initializeDisplay()

    # helper function to initialize instances with same init conditions
    # without randomizing ball or paddle positions
    def _hard_reset(self):

        ball = self.balls[0]
        paddle_learner = self.paddles[0]
        paddle_bot = self.paddles[1]

        BALL_HEIGHT = ball.height
        BALL_WIDTH = ball.width

        PADDLE_BUFFER = paddle_learner.buffer
        PADDLE_WIDTH = paddle_learner.width
        PADDLE_HEIGHT = paddle_learner.height

        WINDOW_WIDTH = self.SCREEN_WIDTH
        WINDOW_HEIGHT = self.SCREEN_HEIGHT

        # initialize the x and y position in the middle of the screen
        ball.x = WINDOW_WIDTH // 2 - BALL_WIDTH // 2
        ball.y = (WINDOW_HEIGHT - BALL_HEIGHT) // 2

        # initialize a direction (down, to the right - south east diagonal)
        ball.x_direction = BALL_X_SPEED
        ball.y_direction = BALL_Y_SPEED

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

        # update paddles (eventuelly, they need actors / agents / controllers )
        for i in range(self.num_paddles):
            self.paddles[i].update(actions[i])


    def initializeDisplay(self):
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

    def render(self):
        # for each frame, calls the event queue, like if the main window needs to be repainted
        pygame.event.pump()
        # make the background black
        self.screen.fill(BLACK)
        # render the entities (balls, paddles)
        self.renderEntities()
        # copy the pixels from game surface to a 3D array. Can be used as input
        # for the network
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # updates the window
        pygame.display.flip()
        # return surface image data
        return image_data

    # update the game state (has to take action as arg?)
    def step(self, actions):

        self.updateEntities(actions)
        reward, hits, hit_reward = self.detect_events()

        image_data = self.render()

        return reward, hits, image_data, hit_reward

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
        
        # when ball rewards only
        # hit_reward =  -1.0
            
        
        # learner_score = 0
        # opponent_score = 0
        # paddle_score = [0, 0] # score for learner and opponent
        # paddle : learning agent (or human control) and bot opponent (or human control)

        # check for collision with paddles
        # here, paddle_learner_XPos = PADDLE_BUFFER + PADDLE_WIDTH can be used, computed beforehand
        # checks for a collision, if the ball hits the Player side - the learning agent
        if (ballXPos <= paddle_learner_XPos + PADDLE_WIDTH and ballYPos + BALL_HEIGHT >= paddle_learner_YPos and ballYPos - BALL_HEIGHT <= paddle_learner_YPos + PADDLE_HEIGHT and ballXDirection < 0.0):
            # case : ball caught by the learning agent paddle (by convention: on the left side)
            # reward situation for classic for case of ball catching:
            # - either no reward (harder case, no intermediate information)
            # - or small reward to indicate that catching ball is a good thing toward winning
            # (main principle : keep reward sensation local to the agent ?)
            
            
            
            # when taking into account score reward: 
            # hit_reward =  0            
            # score_reward = 0

            # when considering both (in this case, when opponent misses, score_reward should be higher than hit_reward)
            # hit_reward = 1.0            
            # score_reward = 0
            
            # when ball rewards only
            hit_reward = 1.0
            
            
            
            # switches directions (ATT: make sure this has effect on the original object)
            ball.x_direction *= -1.0
            ball.y_direction += GAMMA * paddle_learner.vy
            norm = np.sqrt(ball.x_direction**2.0 + ball.y_direction**2.0)
            ball.x_direction = (ball.x_direction / norm) * BALL_SPEED
            ball.y_direction = (ball.y_direction / norm) * BALL_SPEED
            if np.abs(ball.x_direction) < EPSILON or np.abs(ball.y_direction) < EPSILON:
                ball.x_direction = np.sign(ball.x_direction) * random.uniform(EPSILON, np.sqrt(BALL_SPEED**2.0 - EPSILON**2.0))
                ball.y_direction = np.sign(ball.x_direction) * np.sqrt(BALL_SPEED**2.0 - ball.x_direction**2.0)
            #  Player returning the ball  gets a reward for simplified state based learning, or to speed it up - kind of cheat (although it can be argued that sensing the ball corresponds to sensing a raw reward)
            # score = 10.0 # for simplified state based
            # for image based state learning, no reward provided (but can be also employed)
            #

            # hit counter for learner paddle
            hits[0] += 1


            # ---- here depending on the task ---------------
            # if task is to avoid, negative score for the learning agent
            # self.score[0] -= 1

            # --- learner catching the ball ---
            if self.taskActive[0] :
                # -- classical pong
                self.total_hits[0][0] += 1
                # eventualy, help positive reward for learner catching the ball
                # reward[0][0] += 1
            elif self.taskActive[1] :
                # -- avoidance pong
                self.total_hits[1][0] += 1
                # avoidance pong, learner collided with the ball
                # learner could not avoid the ball, punish the learner?
                # reward for the opponent
                reward[0][1] += -1
                reward[1][0] += 1
                self.total_reward[1][0][1] += -1
                self.total_reward[1][1][0] += 1

            # can be outcommented if no visual debugging - signaling a loss by ball color -  is needed
            # NewBallColor = BLUE
            # return reward
        # if player misses/avoids the ball
        elif (ballXPos <= 0):
            # case : ball missed by the learning agent paddle (by convention: on the left side)
            # reward situation for classic for case of missing the ball:
            # - punishment, negative reward - loosing situation (main principle : keep reward sensation local to the agent ?)
            # (although the agent anyway perceives the whole field; locality would make sense if one would require the agent's body is the paddle and every sensory input should be treated as coming through the paddle's "body")

            # when taking into account score reward: 
            # hit_reward =  0
            
            # score_reward = -1.0
            
            # when ball rewards only
            hit_reward =  -1.0
            
            
            # negative score
            # bump from the wall, reverse direction
            ball.x_direction *= -1.0
            # Learning agent misses the ball, so negative score reward
            # score = -10.0 # for simplified state based

            # ---- here depending on the task ---------------
            # if task is to hit, negative score for the learning agent
            # if the task is to avoid, everything is okay, neutral

            # self.score[0] -= 1
            # learner_score = -1.0  # for image based state learning
            # negative reward for learner agent for missing the ball

            # --- learner missing the ball ---
            if self.taskActive[0] :
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
            elif self.taskActive[1] :
                # learner missed the ball : avoidance pong
                pass

            # positive reward for opponent for scoring against the learner


            # can be outcommented if no visual debugging - signaling a loss by ball color -  is needed
            # NewBallColor = RED

            # ----- after missing a ball : can be considered as end of game episode,
            # - either reset the game to random or specified initial conditions
            # - or continue using the ball just bumping off from the screen edge
            # any reason one should prefer reset over non-reset or other way around?

            # resetting the whole game states after ball miss, marking an end of a game episode (loosing)
            # [ballXPos, ballYPos, ballXDirection, ballYDirection] = self.ResetAfterBallMiss()
            # drawPaddle1(self.paddle1YPos)
            # drawPaddle2(self.paddle2YPos)
            # return reward

        # check if ball hits the opponent player
        if (ballXPos >= paddle_bot_XPos and ballYPos + BALL_HEIGHT >= paddle_bot_YPos and ballYPos - BALL_HEIGHT <= paddle_bot_YPos + PADDLE_HEIGHT and ballXDirection >= 0.0):
            # case : ball caught by the ** opponent ** agent paddle (here by convention: on the right side)
            # reward situation for classic for case of ball catching by opponent:
            # - no reward for the learning agent (a smaller reward should be provided to opponent in case it learns, too)
            # - an interesting intermediate modification may be a negative reward for the learning agent, which should push forward strategies to hit the ball back in the way to make it hard for the opponent to catch it

            # switch directions
            ball.x_direction *= -1.0
            ball.y_direction += GAMMA * paddle_learner.vy
            norm = np.sqrt(ball.x_direction ** 2.0 + ball.y_direction ** 2.0)
            ball.x_direction = (ball.x_direction / norm) * BALL_SPEED
            ball.y_direction = (ball.y_direction / norm) * BALL_SPEED
            if np.abs(ball.x_direction) < EPSILON or np.abs(ball.y_direction) < EPSILON:
                ball.x_direction = np.sign(ball.x_direction) * random.uniform(EPSILON, np.sqrt(BALL_SPEED**2.0 - EPSILON**2.0))
                ball.y_direction = np.sign(ball.x_direction) * np.sqrt(BALL_SPEED**2.0 - ball.x_direction**2.0)
            # can be outcommented if no visual debugging - signaling a loss by ball color -  is needed
            # NewBallColor = WHITE

            # hit counter for opponent paddle
            hits[1] += 1

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
            if self.taskActive[0] :
                # total_hits IDs : [task_ID][paddle_ID]
                self.total_hits[0][1] += 1
            elif self.taskActive[1] :
                self.total_hits[1][1] += 1
                # opponent could not avoid the ball, punish the opponent?
                # positive reward for the learner? yes, for the symmetry
                reward[0][0] += 1
                reward[1][1] += -1
                self.total_reward[1][0][0] += 1
                self.total_reward[1][1][1] += -1


            # return reward

        # opponent missing the ball
        elif (ballXPos >= WINDOW_WIDTH - BALL_WIDTH):
            # positive reward for the learning agent (actually, against the local reward processing principle, as ball miss is happening at the opponent)
            #

            
            # when taking into account score reward: 
            # hit_reward =  0            
            # score_reward = 1.0

            # when considering both (in this case, when opponent misses, score_reward should be higher than hit_reward)
            # hit_reward = 0.0            
            # score_reward = 100.0
            
            # when ball rewards only
            hit_reward = 0
                        
            
            # bump from the wall
            ball.x_direction *= -1.0

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
            if self.taskActive[0] :
                # opponent missed the ball : classic pong
                # positive reward for the learner
                reward[0][0] += 1
                # negative reward for the opponent
                reward[1][1] += -1
                self.total_reward[0][0][0] += 1
                self.total_reward[0][1][1] += -1
            elif self.taskActive[1] :
                # opponent missed the ball : avoidance pong
                pass

            # positive reward for the learner for scoring against the opponent
            # reward[0][0] += 1
            # negative reward for the opponent for missing the ball
            # reward[1][1] += -1
            # self.game_score[0] += 1


            # resetting the whole game states after ball miss, marking an end of a game episode (loosing)
            # [ballXPos, ballYPos, ballXDirection, ballYDirection] = self.ResetAfterBallMiss()
            # drawPaddle1(self.paddle1YPos)
            # drawPaddle2(self.paddle2YPos)

            # return reward

        # if the ball hits the top screen edge - move down
        if (ballYPos <= 0):
            ball.y = 0
            ball.y_direction *= -1.0 # downwards direction from top to bottom is positive, 1
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

        # detect and count ball crossing the midline
        if (ballXPos == ((WINDOW_WIDTH - BALL_WIDTH) // 2)) and (not ball.x_direction == 0) :
            if self.taskActive[0]:
                self.ball_cross_counter[0] += 1
            elif self.taskActive[1]:
                self.ball_cross_counter[1] += 1

            # resetable task counter for running within one task cycle
            # counting here ball exchanges
            self.task_iter_counter += 1
            # test :
            # print 'Ball crossed the field count : ', self.ball_cross_counter
        return reward, hits, hit_reward
