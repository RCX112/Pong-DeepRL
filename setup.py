from setuptools import setup, find_packages


setup(name='pong_deeprl',
        version='2.0.0',
        url='https://gitlab.version.fz-juelich.de/azzam1/pong_deeprl',
        license='GPLv3',
        author='Josef Amin Azzam',
        author_email='j.azzam@fz-juelich.de',
        description='Deep Reinforcement Learning Library for the ATARI Videogame Pong in 2D and 3D.',
      
        python_requires='>=3.6',
      
        entry_points={
            'console_scripts': [
                'deep_rllib = src.deeprl_lib.rllib.jupong2d_ppo:start',
                'plot_rllib = src.deeprl_lib.rllib.jupong2d_plot_ppo_data:start',
                'deep_stablebl = src.deeprl_lib.stablebaselines.jupong2d_ppo2:start',
                'plot_stablebl = src.deeprl_lib.stablebaselines.jupong2d_plot_ppo2_data:start',
            ],
        },
      )
