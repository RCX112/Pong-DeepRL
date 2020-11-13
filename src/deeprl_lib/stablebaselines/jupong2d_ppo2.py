"""
This file contains an example to run the Gym-Environment JuPong2D with a Proximal Policy
Optimization algorithm (PPO2) from the Python library Stable Baselines.
"""
import gym, gym_pong
import argparse, os, csv, sys
import numpy as np
from threading import Thread

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

class JuPong2D_PPO2:
    """
    This class is responsible for the training of the Gym-Environment JuPong2D by using different parameters. The
    current choosen parameters are

    - paddle length

    - paddle speed

    - ball speed

    which can vary by a given scale factor.
    """

    def __init__(self, env, output, train_steps, total_time_steps, session, paddle_length_factor=None, paddle_speed_factor=None,
                 ball_speed_factor=None):
        """
        The constructor of the class 'JuPong2D_PPO2' creates a vectorized Gym-Environment with a specific parameter set.
        The neuronal networks will be saved in the output folder after every 'total_time_steps' step. For a more
        accurate training, a parameter 'session' will be used to train the same model multiple times.
        :param env: The Gym-Environment to load
        :param output: The output folder for the neuronal networks
        :param total_time_steps: Training duration before saving
        :param session: Session-ID for a specific training configuration
        :param paddle_length_factor: Factor for the paddle length
        :param paddle_speed_factor: Factor for the paddle speed
        :param ball_speed_factor: Factor for the ball speed
        """
        self.train_steps = train_steps
        self.total_time_steps = total_time_steps
        self.env_name = env
        self.session = session
        self.env = make_vec_env(self.env_name, n_envs=4)
        self.output = output
        self.paddle_length_factor = paddle_length_factor
        self.paddle_speed_factor = paddle_speed_factor
        self.ball_speed_factor = ball_speed_factor
        self.save_name = "stablebl_ppo2_save"
        self.create_save_folder()
        self.make_save_path()
        self.create_model()

    def create_save_folder(self):
        """
        Creates a folder tree which depends on the configurations in the constructor for saving the neuronal networks.
        """
        absolute_output = os.path.abspath(self.output).replace("\\", "/")
        if self.paddle_length_factor is not None:
            self.save_folder = f"{absolute_output}/{self.env_name}/PaddleLength_" \
                               f"{self.paddle_length_factor}/session{self.session}"
        elif self.paddle_speed_factor is not None:
            self.save_folder = f"{absolute_output}/{self.env_name}/PaddleSpeed_" \
                               f"{self.paddle_speed_factor}/session{self.session}"
        elif self.ball_speed_factor is not None:
            self.save_folder = f"{absolute_output}/{self.env_name}/BallSpeed_" \
                               f"{self.ball_speed_factor}/session{self.session}"
        else:
            self.save_folder = f"{absolute_output}/{self.env_name}/StandardEnv/session{self.session}"
        tmp_folder = self.save_folder

        folder_tree = []
        while True:
            if not os.path.exists(self.save_folder):
                folder_tree.insert(0, self.save_folder)
                self.save_folder = self.save_folder[:self.save_folder.rindex("/")]
            else:
                self.save_folder = tmp_folder
                break
        for folder in folder_tree:
            os.mkdir(folder)

    def make_save_path(self):
        """
        Creates the path of the neuronal network which will be saved there.
        """
        if self.paddle_length_factor is not None:
            for gym_env in self.env.envs:
                gym_env.scale_paddle_height(self.paddle_length_factor)
            self.save_path = f"{self.save_folder}/{self.save_name}_paddle_length_{self.paddle_length_factor}"
        elif self.paddle_speed_factor is not None:
            for gym_env in self.env.envs:
                gym_env.scale_paddle_vel(self.paddle_speed_factor)
            self.save_path = f"{self.save_folder}/{self.save_name}_paddle_speed_{self.paddle_speed_factor}"
        elif self.ball_speed_factor is not None:
            for gym_env in self.env.envs:
                gym_env.scale_ball_velocity(self.ball_speed_factor)
            self.save_path = f"{self.save_folder}/{self.save_name}_ball_speed_{self.ball_speed_factor}"
        else:
            self.save_path = f"{self.save_folder}/{self.save_name}"

    def create_model(self):
        """
        Loads the neuronal network in the given save path . If it hasn't be saved yet a new model will be created.
        """
        try:
            self.model = PPO2.load(self.save_path)
            self.model.set_env(self.env)
            print("Loading of the latest model successful!")
        except:
            print("Creating new model...")
            self.model = PPO2(CnnPolicy, self.env, verbose=1)

    def start_training(self):
        """
        Starts the training of the model in the given Gym-Environment and saves it every 'total_time_steps' step.
        """
        i = 0
        for _ in range(self.train_steps):
            print(f"Start Training Step {i + 1}")
            self.model.learn(total_timesteps=self.total_time_steps)
            self.model.save(self.save_path)
            print(f"Finished Training Step {i + 1}")
            i += 1


class JuPong2D_PPO2_Play:
    """
    The class 'JuPong2D_PPO2_Play' analyses the quality of a model which is characterized by the parameters of the Gym-
    Environment JuPong2D.
    """
    def __init__(self, env, output, session, play_steps, paddle_length_factor=None, paddle_speed_factor=None,
                 ball_speed_factor=None):
        """
        The constructor of the class 'JuPong2D_PPO2_Play' creates a test-setup for the model and a varying
        Gym-Environment. It loads a neuronal network from the output path and creates the Gym-Environment from the given
        parameters.
        :param env: The Gym-Environment to load
        :param output: The output folder for the neuronal networks
        :param session: Session-ID for a specific training configuration
        :param play_steps: Number of playing steps
        :param paddle_length_factor: Factor for the paddle length
        :param paddle_speed_factor: Factor for the paddle speed
        :param ball_speed_factor: Factor for the ball speed
        """
        self.env_name = env
        self.session = session
        self.play_steps = play_steps
        self.output = output
        self.paddle_length_factor = paddle_length_factor
        self.paddle_speed_factor = paddle_speed_factor
        self.ball_speed_factor = ball_speed_factor
        self.save_name = "stablebl_ppo2_save"
        self.create_save_folder()
        self.create_model()
        self.return_arr = None
        self.scale_factor_arr = None
        self.save_file = None

    def create_save_folder(self):
        """
        Creates the path to the neuronal network.
        """
        if self.paddle_length_factor is not None:
            self.save_folder = f"{self.output}/{self.env_name}/PaddleLength_" \
                               f"{self.paddle_length_factor}/session{self.session}"
        elif self.paddle_speed_factor is not None:
            self.save_folder = f"{self.output}/{self.env_name}/PaddleSpeed_" \
                               f"{self.paddle_speed_factor}/session{self.session}"
        elif self.ball_speed_factor is not None:
            self.save_folder = f"{self.output}/{self.env_name}/BallSpeed_" \
                               f"{self.ball_speed_factor}/session{self.session}"
        else:
            print("Keinen Parameter ausgewaehlt")
            sys.exit()

    def create_model(self):
        """
        Loads the neuronal network and sets the Gym-Environment to play in.
        """
        env = make_vec_env(self.env_name, n_envs=4)

        if self.paddle_length_factor is not None:
            for gym_env in env.envs:
                gym_env.scale_paddle_height(self.paddle_length_factor)
            save_path = f"{self.save_folder}/{self.save_name}_paddle_length_{self.paddle_length_factor}"
        elif self.paddle_speed_factor is not None:
            for gym_env in env.envs:
                gym_env.scale_paddle_vel(self.paddle_speed_factor)
            save_path = f"{self.save_folder}/{self.save_name}_paddle_speed_{self.paddle_speed_factor}"
        elif self.ball_speed_factor is not None:
            for gym_env in env.envs:
                gym_env.scale_ball_velocity(self.ball_speed_factor)
            save_path = f"{self.save_folder}/{self.save_name}_ball_speed_{self.ball_speed_factor}"
        else:
            print("Keinen Parameter ausgewaehlt")
            sys.exit()

        self.model = PPO2.load(os.path.abspath(save_path))
        self.model.set_env(env)

    def process_environment(self, ind, paddle_length=None, paddle_speed=None, ball_speed=None):
        """
        This method will be executed by multiple threads. It measures the quality of a neuronal network by analyzing
        different parameter values of the Gym-Environment JuPong2D. The results are mean-return-values, which will be
        saved in a csv-file.
        :param ind: Thread index for a scale factor
        :param paddle_length: Factor for the paddle length
        :param paddle_speed: Factor for the paddle speed
        :param ball_speed: Factor for the ball speed
        """
        env = make_vec_env(self.env_name, n_envs=4)

        if paddle_length is not None:
            for gym_env in env.envs:
                gym_env.scale_paddle_height(paddle_length)
            print(f"Paddle Length {paddle_length}")
        elif paddle_speed is not None:
            for gym_env in env.envs:
                gym_env.scale_paddle_vel(paddle_speed)
            print(f"Paddle Speed {paddle_speed}")
        elif ball_speed is not None:
            for gym_env in env.envs:
                gym_env.scale_ball_velocity(ball_speed)
            print(f"Ball Speed {ball_speed}")
        else:
            print("Kein Parameter gesetzt.")
            return

        obs = env.reset()
        return_vals = np.array([0.0, 0.0, 0.0, 0.0])
        return_val_arr = []
        done_cnt = 0
        
        for _ in range(self.play_steps):
            while True:
                action, _states = self.model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                return_vals += rewards

                for i, done in enumerate(dones):
                    if done:
                        done_cnt += 1
                        return_val_arr.append(return_vals[i])
                        return_vals[i] = 0.0
                        self.return_arr[ind] = np.mean(return_val_arr)

                        print(self.return_arr)
                        with open(self.save_file, 'w') as my_file:
                            writer = csv.writer(my_file)
                            writer.writerow(self.scale_factor_arr)
                            writer.writerow(self.return_arr)
                            
                if done_cnt >= 4:
                    break
            

    def create_data(self):
        """
        This method creates the analysis setup for a given model. It uses threading to test the model in different
        Gym-Environments, which differ on the parameter-values.
        """
        num_left_right = None
        threads = []
        self.scale_factor_arr = []

        if self.paddle_length_factor is not None:
            step_size = 0.2
            num_left_right = [-2, 4]
            self.save_file = f"{self.save_folder}/paddle_length_{self.paddle_length_factor}.csv"
            for i in range(num_left_right[0], num_left_right[1] + 1):
                self.scale_factor_arr.append(self.paddle_length_factor + i * step_size)
        elif self.paddle_speed_factor is not None:
            step_size = 0.2
            num_left_right = [-3, 3]
            self.save_file = f"{self.save_folder}/paddle_speed_{self.paddle_speed_factor}.csv"
            for i in range(num_left_right[0], num_left_right[1] + 1):
                self.scale_factor_arr.append(self.paddle_speed_factor + i * step_size)
        elif self.ball_speed_factor is not None:
            step_size = 0.2
            num_left_right = [-3, 3]
            self.save_file = f"{self.save_folder}/ball_speed_{self.ball_speed_factor}.csv"
            for i in range(num_left_right[0], num_left_right[1] + 1):
                self.scale_factor_arr.append(self.ball_speed_factor + i * step_size)

        else:
            print("No parameter chosen for test")
            sys.exit()
            
        self.return_arr = np.zeros(len(self.scale_factor_arr))
        for i, sf in enumerate(self.scale_factor_arr):
            t = Thread(target=self.process_environment, args=(i, sf))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        print("Testing finished\n\n\n\n\n\n\n\n\n\n\n\n")


def start():
    """
    Start method of the training and testing setup in Stable Baselines for JuPPong2D.
    Usage: deep_stablebl [-h] [--pl PL] [--ps PS] [--bs BS] [--env ENV]
                         [--tts TTS] [--train-steps TRAIN_STEPS]
                         [--session SESSION] [--play] [--play-steps PLAY_STEPS]
                         output

    positional arguments:
        output                Path to the results folder of the given environment.

    optional arguments:
        -h, --help            show this help message and exit
        --pl PL               Factor of the paddle-length.
        --ps PS               Factor of the paddle-speed.
        --bs BS               Factor of the ball-speed.
        --env ENV             The Gym-Environment to load
        --tts TTS             Total timesteps.
        --train-steps TRAIN_STEPS
                            Number of training steps.
        --session SESSION     Training session for a more precise analysis
        --play                Start the parameter analysis by testing the
                            environment
        --play-steps PLAY_STEPS
                            Number of playing steps
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help='Path to the results folder of the given environment.')
    parser.add_argument("--pl", type=float, default=None, help="Factor of the paddle-length.")
    parser.add_argument("--ps", type=float, default=None, help="Factor of the paddle-speed.")
    parser.add_argument("--bs", type=float, default=None, help="Factor of the ball-speed.")
    parser.add_argument("--env", type=str, default="jupong2d-headless-0.4-v3", help="The Gym-Environment to load")
    parser.add_argument("--tts", type=int, default=50_000, help='Total timesteps per training step')
    parser.add_argument("--train-steps", type=int, default=10, help='Number of training steps')
    parser.add_argument("--session", type=int, default=1, help="Training session for a more precise analysis")
    parser.add_argument("--play", action="store_true", help="Start the parameter analysis by testing the environment")
    parser.add_argument("--play-steps", type=int, default=5, help="Number of playing steps")
    args = parser.parse_args()


    if not args.play:
        agent = JuPong2D_PPO2(args.env, args.output, args.train_steps, args.tts, args.session, args.pl, args.ps, args.bs)
        agent.start_training()
    else:
        agent = JuPong2D_PPO2_Play(args.env, args.output, args.session, args.play_steps, args.pl, args.ps, args.bs)
        agent.create_data()


if __name__ == "__main__":
    start()
