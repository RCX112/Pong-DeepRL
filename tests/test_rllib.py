"""
This python file contains a TestSuite for the Pong DeepRL project.
"""
import unittest
import sys
from src.deeprl_lib.rllib.jupong2d_ppo import *
from src.deeprl_lib.rllib.jupong2d_plot_ppo_data import *

class TestRllib(unittest.TestCase):
    """
    Testcase for the library RLlib.
    """
    def __init__(self, *args, **kwargs):
        super(TestRllib, self).__init__(*args, **kwargs)
        self.run_alg = "PPO"
        self.num_cpus = 0
        self.env_name = "jupong2d"
        self.num_workers = 3
        self.env_per_worker = 5
        self.stop_reward = None
        self.stop_iters = 5
        self.stop_timesteps = None
        
        self.output_folder = "test_rllib"
        self.paddle_length_factor = 0.5
        self.session = 1
        self.checkpoint_freq = 5

        self.restore = True
        self.play = True
        self.play_steps = 1
        
    def setUp(self):
        pass

    def runTest(self):
        print("Running tests of the class TestRllib")
        self.test_configurations()
        self.test_training()
        self.test_playing()
        self.test_ploting()
            
    def test_configurations(self):
        train_runner = RunRLlib(self.output_folder, self.num_cpus, self.env_name, self.paddle_length_factor,                                                       session=self.session, checkpoint_frequency=self.checkpoint_freq, stop_iters=self.stop_iters,                                       restore=self.restore)
        self.assertTrue(train_runner.stop is not None)
        
    def test_training(self):
        train_runner = RunRLlib(self.output_folder, self.num_cpus, self.env_name, self.paddle_length_factor,                                                       session=self.session, checkpoint_frequency=self.checkpoint_freq, stop_iters=self.stop_iters,                                       restore=self.restore)
        train_runner.start()
    
    def test_playing(self):
        play_runner = RunRLlib(self.output_folder, self.num_cpus, self.env_name, self.paddle_length_factor,                                                       session=self.session, play=self.play, play_steps=self.play_steps)
        play_runner.start()
    
    def test_ploting(self):
        ploter = JuPong2D_PPO_Plot(self.output_folder)
        ploter.plot_paddle_length()
        

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestRllib())
    unittest.TextTestRunner().run(suite)
