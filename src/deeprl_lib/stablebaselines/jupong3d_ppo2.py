import gym, gym_pong
import argparse
import gym_unrealcv
import os

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

class JuPong3D_PPO2:
    """
    This class is responsible for the training of the Gym-Environment JuPong2D by using different parameters. The
    current choosen parameters are

    - paddle length

    which can vary by a given scale factor.
    """

    def __init__(self, env, output, train_steps, total_time_steps, session, paddle_length_factor=None):
        """
        The constructor of the class 'JuPong2D_PPO2' creates a vectorized Gym-Environment with a specific parameter set.
        The neuronal networks will be saved in the output folder after every 'total_time_steps' step. For a more
        accurate training, a parameter 'session' will be used to train the same model multiple times.
        :param env: The Gym-Environment to load
        :param output: The output folder for the neuronal networks
        :param total_time_steps: Training duration before saving
        :param session: Session-ID for a specific training configuration
        :param paddle_length_factor: Factor for the paddle length
        """
        self.train_steps = train_steps
        self.total_time_steps = total_time_steps
        self.env_name = env
        self.session = session
        self.env = make_vec_env(self.env_name, n_envs=1)
        self.output = output
        self.paddle_length_factor = paddle_length_factor
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


def start():
    """
    Start method of the training and testing setup in Stable Baselines for JuPPong2D.
    Usage: deep_stablebl [-h] [--pl PL] [--env ENV]
                         [--tts TTS] [--train-steps TRAIN_STEPS]
                         [--session SESSION] [--play] [--play-steps PLAY_STEPS]
                         output

    positional arguments:
        output                Path to the results folder of the given environment.

    optional arguments:
        -h, --help            show this help message and exit
        --pl PL               Factor of the paddle-length.
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
    parser.add_argument("--env", type=str, default="jupong-3D-Windows-v0", help="The Gym-Environment to load")
    parser.add_argument("--tts", type=int, default=50000, help='Total timesteps per training step')
    parser.add_argument("--train-steps", type=int, default=10, help='Number of training steps')
    parser.add_argument("--session", type=int, default=1, help="Training session for a more precise analysis")
    parser.add_argument("--play", action="store_true", help="Start the parameter analysis by testing the environment")
    parser.add_argument("--play-steps", type=int, default=5, help="Number of playing steps")
    args = parser.parse_args()

    if not args.play:
        agent = JuPong3D_PPO2(args.env, args.output, args.train_steps, args.tts, args.session, args.pl)
        agent.start_training()
    else:
        pass


if __name__ == "__main__":
    start()
