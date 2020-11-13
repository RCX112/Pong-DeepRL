import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from pong_basis import EnvPongDraft_Surface_Headless, EnvPongDraft

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled(
            "{}. (HINT: you can install Atari dependencies by running "
            "'pip install gym[atari]'.)".format(e))


def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size), dtype=np.uint8)
    ale.getRAM(ram)
    return ram


class JuPong2D(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            game='pong',
            mode=None,
            difficulty=None,
            obs_type='ram',
            frameskip=(2, 5),
            repeat_action_probability=0.,
            full_action_space=False, 
            render_screen = False,
            screen_scale = None):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(
                self,
                game,
                mode,
                difficulty,
                obs_type,
                frameskip,
                repeat_action_probability)
        assert obs_type in ('ram', 'image')

        self.game = game
        self.game_path = atari_py.get_game_path(game)
        self.game_mode = mode
        self.game_difficulty = difficulty

        if not os.path.exists(self.game_path):
            msg = 'You asked for game %s but path %s does not exist'
            raise IOError(msg % (game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        #self.ale = atari_py.ALEInterface()
        self.ale = EnvPongDraft_Surface_Headless.EnvPong(render_screen = render_screen, screen_scale = screen_scale)
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), \
                "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat(
                'repeat_action_probability'.encode('utf-8'),
                repeat_action_probability)

        self.seed()

        self._action_set = (self.ale.getLegalActionSet() if full_action_space
                            else self.ale.getMinimalActionSet())
        self.action_space = spaces.Discrete(len(self._action_set))
        #self.action_space = spaces.Box(
         #   low=0,
          #  high=1, shape=(1,),
           # dtype=np.float32
        #)
        (screen_width, screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)
        self.ale.loadROM(self.game_path)

        if self.game_mode is not None:
            modes = self.ale.getAvailableModes()

            assert self.game_mode in modes, (
                "Invalid game mode \"{}\" for game {}.\nAvailable modes are: {}"
            ).format(self.game_mode, self.game, modes)
            self.ale.setMode(self.game_mode)

        if self.game_difficulty is not None:
            difficulties = self.ale.getAvailableDifficulties()

            assert self.game_difficulty in difficulties, (
                "Invalid game difficulty \"{}\" for game {}.\nAvailable difficulties are: {}"
            ).format(self.game_difficulty, self.game, difficulties)
            self.ale.setDifficulty(self.game_difficulty)

        return [seed1, seed2]

    def step(self, a):
        #action = int(np.round(a[0]))
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.ale.act(action)
        ob = self._get_obs()

        return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def _get_image(self):
        return self.ale.getScreenRGB2()

    def _get_ram(self):
        return to_ram(self.ale)
    
    def change_ball_velocity(self, ball_vmin, ball_vmax):
        ball = self.ale.get_ball()
        ball.set_vmin(ball_vmin)
        ball.set_vmax(ball_vmax)
        
    def scale_ball_velocity(self, scale_factor):
        self.ale.scale_ball_velocity(scale_factor)
        
    def set_ball_color(self, color):
        self.ale.set_ball_color(color)

    def scale_paddle_height(self, scale_factor):
        self.ale.scale_paddles(scale_factor)

    def scale_paddle_vel(self, scale_factor):
        self.ale.scale_paddle_vel(scale_factor)

    def get_ball_hits(self):
        return self.ale.get_ball_hits()
    
    def get_ball_misses(self):
        return self.ale.get_ball_misses()
    
    def get_ball_crossing_sides(self):
        return self.ale.get_ball_crossing_sides()
    
    def get_agent_ball_angles(self):
        return self.ale.get_agent_ball_angles()
    
    def get_cpu_ball_angles(self):
        return self.ale.get_cpu_ball_angles()
    
    def get_agent_ball_dir_and_vel(self):
        return self.ale.get_agent_ball_dir_and_vel()



    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == 'ram':
            return self._get_ram()
        elif self._obs_type == 'image':
            img = self._get_image()
        return img

    # return: (states, observations)
    def reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)


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
