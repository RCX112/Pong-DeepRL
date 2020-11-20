"""
This file contains an example to run the Gym-Environment JuPong2D with a Proximal Policy
Optimization algorithm from the Python library RLlib.

You can find another good example here: https://github.com/ray-project/ray/blob/master/rllib/examples/cartpole_lstm.py
"""
import argparse
import gym
import csv
import numpy as np
import platform
import os

from pong_basis import EnvPongDraft_Surface_Headless
from gym.utils import seeding
from gym import spaces
from scipy.ndimage import zoom
from threading import Thread

import ray
from ray.tune.registry import register_env
from ray.rllib.utils.test_utils import check_learning_achieved
import ray.rllib.agents.ppo as ppo
from ray import tune


class JuPong2D(gym.Env):
    """
    This class is a simple version of the JuPong2D-Environment. The size of the images for every step is 42 x 42.
    More details can be found in the class 'EnvPongDraft_Surface_Headless'.
    """
    def __init__(self, env_config, paddle_length_factor):
        """
        This method creates a Pong game from the class 'EnvPongDraft_Surface_Headless' and scales the paddle length
        with a given factor. The observation space is 42 x 42 and the agent can use the actions 0 and 1 for moving the
        paddle up and down.
        :param env_config: Optional configuration variable
        """
        self.env_config = env_config
        self.frameskip = (2, 5)
        self.env = EnvPongDraft_Surface_Headless.EnvPong(render_screen = False, screen_scale = 1.0)
        self.scale_paddle_height(paddle_length_factor)
        self.seed()
        self._action_set = self.env.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        screen_width = screen_height = 42
        self.zoom_val = 42 / 400
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width), dtype=np.float)

    def seed(self, seed=None):
        """
        Derives a random seed.
        """
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31

        return [seed1, seed2]

    def step(self, action):
        """
        In this method the agent takes an action a for a random number of steps. The environment evaluates the agent
        with a reward and returns the current observation (42 x 42 image) and an information if the episode is over.
        :param action: Action a of the agent
        :return: Current observation, reward, episode done information and an empty set (can contain extra information)
        """
        reward = 0.0
        paddle_action = self._action_set[action]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.env.act(paddle_action)
        ob = self._get_obs()

        return ob, reward, self.env.game_over(), {}

    def _get_obs(self):
        """
        Gets the observation (42 x 42 image) of the current frame.
        :return: Current observation
        """
        pong_image = self._get_image()
        pong_image = zoom(pong_image[:, :, 0], self.zoom_val)
        return pong_image.astype(np.float)

    def _get_image(self):
        """
        Gets the actual RGB-image of the pong game, which is only black and white.
        :return: 3-dimensional RGB-image of the pong game
        """
        return self.env.getScreenRGB2()

    def reset(self):
        """
        Resets the ball and paddle positions randomly and returns the first observation of the next episode.
        :return: First observation of new episode.
        """
        self.env.reset_game()
        return self._get_obs()

    def scale_paddle_height(self, scale_factor):
        """
        Scales the paddles by a given factor.
        :param scale_factor: Scale factor for the paddle length.
        """
        self.env.scale_paddles(scale_factor)


class RunRLlib:
    def __init__(self, output_folder, num_cpus, env_name, paddle_length, session=1, checkpoint_frequency=10, train_algorithm="PPO"
                 , num_workers=3, env_per_worker=5, stop_reward=None, stop_iters=None, stop_timesteps=None, restore=True, 
                 as_test=False, play=False, play_steps=3):
        self.num_cpus = num_cpus
        self.paddle_length = paddle_length
        self.session = session
        self.checkpoint_freq = checkpoint_frequency
        self.run_alg = train_algorithm
        self.num_workers = num_workers
        self.env_per_worker = env_per_worker
        self.stop_reward = stop_reward
        self.stop_iters = stop_iters
        self.stop_timesteps = stop_timesteps
        self.output_folder = output_folder
        self.restore = restore
        self.play = play
        self.play_steps = play_steps
        self.as_test = as_test
        self.env_name = env_name

        ray.shutdown()
        ray.init(num_cpus=self.num_cpus or None)
        self.register_gym_env(self.env_name, self.paddle_length)

        self.config, self.stop = self.rllib_configurations(self.run_alg, self.env_name, self.num_workers, self.env_per_worker, 
                                     stop_reward=self.stop_reward, stop_iters=self.stop_iters, stop_timesteps=self.stop_timesteps)
        self.save_folder, self.results_path = self.create_result_paths(self.output_folder, self.session, self.paddle_length)

        self.latest_checkpoint_path = None
        self.checkpoint_number = None
        self.handle_checkpoint()
        print(f"Path to latest checkpoint: {self.latest_checkpoint_path} with CP-number {self.checkpoint_number}")

    def start(self):
        if not self.play:
            print("Training the model")
            self.train_model(self.run_alg, self.config, self.stop, self.output_folder, self.checkpoint_freq, self.save_folder,                                 self.latest_checkpoint_path, self.as_test, self.stop_reward)
        else:
            print(f"Testing the model {self.latest_checkpoint_path}")
            self.test_agent(self.env_name, self.config, self.latest_checkpoint_path, self.results_path, self.play_steps,                                      self.paddle_length)
    
    
    def handle_checkpoint(self):
        if os.path.exists(self.results_path) and len(os.listdir(self.results_path)) and (self.restore or self.play):
            self.latest_checkpoint_path, self.checkpoint_number = self.get_latest_checkpoint(self.results_path)
    
    def creation_date(self, path_to_file):
        """
        Try to get the date that a file was created, falling back to when it was
        last modified if that isn't possible.
        """
        if platform.system() == 'Windows':
            return os.path.getctime(path_to_file)

        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


    def get_latest_checkpoint_number(self, folder_path):
        """
        Returns the timestep of the latest checkpoint as an integer.
        :param folder_path: Path to all the checkpoint folders of the environment.
        :return: Timestep of the latest checkpoint.
        """
        num = -1
        cp_folders = os.listdir(folder_path)
        for cp_folder in cp_folders:
            try:
                tmp_num = int(cp_folder.split('_')[-1])
            except ValueError:
                continue
            if tmp_num > num:
                num = tmp_num

        return num


    def process_environment(self, agent, index, scale_factor, results_path, scale_factor_arr, play_steps,
                            paddle_length_factor=None):
        """
        Method for analyzing the paddle length by saving mean return-values of a given scale factor in a csv-file.
        :param paddle_length_factor: Factor for the paddle length.
        :param agent: Agent who calculates an action of a given observation and follows a specific policy.
        :param index: Integer to save the mean return-values in the right way.
        :param scale_factor: Scale factor for the paddle length.
        :param results_path: Directory of the csv-file.
        :param scale_factor_arr: Array of all scale factors to analyze.
        :param play_steps: Number of playing steps
        """
        global return_arr
        env = JuPong2D({}, paddle_length_factor)
        env.scale_paddle_height(scale_factor)
        rewards = []
        for _ in range(play_steps):
            episode_reward = 0
            done = False
            obs = env.reset()
            while not done:
                action = agent.compute_action(obs)
                obs, reward, done, __ = env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            return_arr[index] = np.mean(rewards)
            print(return_arr)
            with open(f"{results_path}/paddle_length_{paddle_length_factor}.csv", 'w') as my_file:
                writer = csv.writer(my_file)
                writer.writerow(scale_factor_arr)
                writer.writerow(return_arr)


    def get_latest_checkpoint(self, results_path):
        """
        Returns the latest checkpoint from a folder path.
        :param results_path: Path to the results folder.
        :return: Path to the latest checkpoint and latest iteration step.
        """
        list_subfolders_with_paths = [f.path for f in os.scandir(results_path) if
                                      f.is_dir() and not f.name.startswith(".")]
        creation_dates = [self.creation_date(folder_path) for folder_path in list_subfolders_with_paths]
        max_ind = creation_dates.index(max(creation_dates))
        newest_folder = list_subfolders_with_paths[max_ind]
        checkpoint_number = self.get_latest_checkpoint_number(newest_folder)
        latest_checkpoint_path = f"{newest_folder}/checkpoint_{checkpoint_number}/checkpoint-{checkpoint_number}" if checkpoint_number >= 0 else None

        return latest_checkpoint_path, checkpoint_number


    def train_model(self, run, config, stop, output_folder, checkpoint_freq, save_folder, latest_checkpoint_path, as_test,                             stop_reward):
        """
        Method for training the model, where the tune command of rllib is used.
        :param run: Deep Reinforcement Learning algorithm.
        :param config: Training configuation.
        :param stop: Stop condition during the training.
        :param output_folder: Path where the checkpoints are saved.
        :param checkpoint_freq: Frequency of saving the next checkpoint.
        :param save_folder: Detailed save path of the checkpoints.
        :param latest_checkpoint_path: Path to the latest checkpoint to continue the training. If it is None, a new session will be created.
        :param as_test: Boolean to test, if the training was successful.
        :param stop_reward: After the training, it will be checked, if the 'stop_reward' is reachable.
        """
        if stop is None:
            raise Exception("No stop condition was set for the training! You need at least one.")
        results = tune.run(run, config=config, stop=stop, checkpoint_at_end=True, local_dir=output_folder,
                           checkpoint_freq=checkpoint_freq, name=save_folder, restore=latest_checkpoint_path)

        if as_test:
            check_learning_achieved(results, stop_reward)
        ray.shutdown()


    def create_result_paths(self, output_folder, session, paddle_length):
        """
        Creates result paths.
        :param output_folder: Path to the root folder of the training results.
        :param session: Training session.
        :param paddle_length: Factor for the paddle length.
        :return: Returns the save folder path and the path from the output folder to the save folder.
        """
        save_folder = f"JuPong2D_PaddleLength_{paddle_length}/session{session}"
        results_path = f"{output_folder}/{save_folder}"
        return save_folder, results_path


    def register_gym_env(self, env_name, paddle_length):
        """
        Here the Gym-Environment JuPong2D with a specific parameter configuration will be registered.
        :param paddle_length: Factor for the paddle length.
        """
        def env_creator(env_config):
            return JuPong2D(env_config, paddle_length)
        register_env(env_name, env_creator)


    def rllib_configurations(self, run, env_name, num_workers, env_per_workers, stop_reward=None, stop_iters=None,                                              stop_timesteps=None):
        """
        This method creates all necessary configurations for the rllib-training. Only one stop condition will be used.
        :param run: Deep Reinforcement Learning algorithm.
        :param num_workers: Number of workers(CPUs) during the training.
        :param env_per_workers: Number of Gym-Environments per worker.
        :param stop_reward: If this reward is being reached the training stops.
        :param stop_iters: After 'stop_iters' iteration the training stops.
        :param stop_timesteps: After 'stop_timesteps' timesteps the training stops.
        :return: Training configuration and stop condition.
        """
        configs = {
            "PPO": {},
            "IMPALA": {
                "num_workers": 2,
                "num_gpus": 0,
                "vf_loss_coeff": 0.01,
            },
            "DQN": {}
        }

        config = dict(
            configs[run], **{
                "env": env_name,
                "model": {
                    "dim": 42
                },
                "framework": "tf",
                "lambda": 0.95,
                "kl_coeff": 0.5,
                "clip_rewards": True,
                "clip_param": 0.1,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
                "train_batch_size": 5000,
                "rollout_fragment_length": 20,
                "sgd_minibatch_size": 500,
                "num_sgd_iter": 10,
                "num_workers": num_workers,
                "num_envs_per_worker": env_per_workers,
                "batch_mode": "truncate_episodes",
                "observation_filter": "NoFilter",
                "vf_share_layers": True,
                "num_gpus": 0
            })

        stop = None
        if stop_iters is not None:
            stop = {"training_iteration": stop_iters}
        elif stop_timesteps is not None:
            stop = {"time_total_s": stop_timesteps}
        elif stop_reward is not None:
            stop = {"episode_reward_mean": stop_reward}

        return config, stop


    def test_agent(self, env_name, config, latest_checkpoint_path, results_path, play_steps, paddle_length):
        """
        Test method for the agent. It calculates the mean return value of a specific model with different environment configurations. Threads are used to create the test results quicker.
        :param config: Configuration of the model.
        :param latest_checkpoint_path: Path to the latest saved checkpoint.
        :param results_path: Path where the test results are saved. 
        :param play_steps: Number of episodes per thread to create a  fairly precisely mean return-value. 
        :param paddle_length: Factor for the paddle length.
        """
        agent = ppo.PPOTrainer(config=config, env=env_name)
        agent.restore(latest_checkpoint_path)
        scale_factor_arr = []

        step_size = 0.2
        num_left_right = [-2, 2]

        for i in range(num_left_right[0], num_left_right[1] + 1):
            scale_factor_arr.append(paddle_length + i * step_size)

        global return_arr
        return_arr = np.zeros(len(scale_factor_arr))

        threads = []
        for i, sf in enumerate(scale_factor_arr):
            t = Thread(target=self.process_environment, args=(agent, i, sf, results_path, scale_factor_arr, play_steps, paddle_length))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        print("Testing finished" + (10 * "\n"))


def register_gym_env(env_name, paddle_length):
    """
    Here the Gym-Environment JuPong2D with a specific parameter configuration will be registered.
    :param paddle_length: Factor for the paddle length.
    """
    def env_creator(env_config):
        return JuPong2D(env_config, paddle_length)
    register_env(env_name, env_creator)        


def start():
    """
    Main method to train or play JuPong2D with RLlib
    usage: deep_rllib [-h] [--run RUN] [--num-cpus NUM_CPUS]
                      [--num-workers NUM_WORKERS]
                      [--env-per-workers ENV_PER_WORKERS] [--as-test]
                      [--use-prev-action-reward] [--stop-iters STOP_ITERS]
                      [--stop-timesteps STOP_TIMESTEPS]
                      [--stop-reward STOP_REWARD] [--pl PL] [--restore]
                      [--checkpoint-freq CHECKPOINT_FREQ] [--play]
                      [--play-steps PLAY_STEPS] [--session SESSION]
                      [--env-name ENV_NAME]
                      output

    positional arguments:
      output                Path to the results folder

    optional arguments:
      -h, --help            show this help message and exit
      --run RUN             Deep RL algorithm to train the agent
      --num-cpus NUM_CPUS   Number of CPUs when training
      --num-workers NUM_WORKERS
                            Number of workers during the training
      --env-per-workers ENV_PER_WORKERS
                            Number of environments per worker
      --as-test             Check if learning was successful
      --use-prev-action-reward
      --stop-iters STOP_ITERS
                            Stop training after specific iterations
      --stop-timesteps STOP_TIMESTEPS
                            Stop training after specific timesteps
      --stop-reward STOP_REWARD
                            Stop training after a specific return-value
      --pl PL               Factor of the paddle length
      --restore             Continue training by restoring the latest checkpoint
      --checkpoint-freq CHECKPOINT_FREQ
                            Frequency of saving the next checkpoint
      --play                Start the parameter analysis by testing the
                            environment
      --play-steps PLAY_STEPS
                            Number of playing steps
      --session SESSION     Training session for a more precise analysis
      --env-name ENV_NAME   Name of the Gym-Environment to register
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Path to the results folder")
    parser.add_argument("--run", type=str, default="PPO", help="Deep RL algorithm to train the agent")
    parser.add_argument("--num-cpus", type=int, default=0, help="Number of CPUs when training")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of workers during the training")
    parser.add_argument("--env-per-workers", type=int, default=5, help="Number of environments per worker")
    parser.add_argument("--as-test", action="store_true", help="Check if learning was successful")
    parser.add_argument("--use-prev-action-reward", action="store_true", help="")
    parser.add_argument("--stop-iters", type=int, default=None, help="Stop training after specific iterations")
    parser.add_argument("--stop-timesteps", type=int, default=None, help="Stop training after specific timesteps")
    parser.add_argument("--stop-reward", type=float, default=None, help="Stop training after a specific return-value")
    parser.add_argument("--pl", type=float, default=None, help="Factor of the paddle length")
    parser.add_argument("--restore", action="store_true", help="Continue training by restoring the latest checkpoint")
    parser.add_argument("--checkpoint-freq", type=int, default=20, help="Frequency of saving the next checkpoint")
    parser.add_argument("--play", action="store_true", help="Start the parameter analysis by testing the environment")
    parser.add_argument("--play-steps", type=int, default=10, help="Number of playing steps")
    parser.add_argument("--session", type=int, default=1, help="Training session for a more precise analysis")
    parser.add_argument("--env-name", type=str, default="jupong2d", help="Name of the Gym-Environment to register")
    args = parser.parse_args()

    runner = RunRLlib(args.output, args.num_cpus, args.env_name, args.pl, session=args.session,                                                         checkpoint_frequency=args.checkpoint_freq, train_algorithm=args.run, num_workers=args.num_workers,                                 env_per_worker=args.env_per_workers, stop_reward=args.stop_reward, stop_iters=args.stop_iters,                                     stop_timesteps=args.stop_timesteps, restore=args.restore, as_test=args.as_test, play=args.play,                                   play_steps=args.play_steps)
    runner.start()

return_arr = None

if __name__ == "__main__":
    start()
