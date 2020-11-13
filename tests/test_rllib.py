"""
TestSuite for the RLlib library.
"""
import unittest
import sys
from deeprl_lib.rllib.jupong2d_ppo import *

# Configurations outside of test class
num_cpus = 0
ray.init(num_cpus=num_cpus or None)

# If you register an environment inside a test class the program will crash
pl = 0.5
register_gym_env(pl)

class TestRllib(unittest.TestCase):
    """
    Testclass for RLlib.
    """
    def __init__(self, *args, **kwargs):
        super(TestRllib, self).__init__(*args, **kwargs)
        self.pl = pl
        self.session = 1
        self.checkpoint_freq = 10
        self.run_alg = "PPO"
        self.num_workers = 16
        self.env_per_worker = 5
        self.stop_iters = 20
        self.stop_timesteps = None
        
        self.config, self.stop = rllib_configurations(self.run_alg, self.num_workers, self.env_per_worker, 
                                     20.0, stop_iters=self.stop_iters, stop_timesteps=self.stop_timesteps)
        self.output_folder = "rllib_test"
        self.save_folder, self.results_path = create_result_paths(self.output_folder, self.session, self.pl)
        self.latest_checkpoint_path = None
        
    def setUp(self):
        pass

    def test_training(self):
        pass


if __name__ == '__main__':
    unittest.main()
