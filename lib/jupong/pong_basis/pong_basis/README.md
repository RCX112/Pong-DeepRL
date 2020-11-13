Code for an example multi-task environment containing simple Pong game dynamics and reward / punishment rules. The rules can be switched during environment execution. In the demonstration, classical pong game (catch the ball - ball contact is rewarded, losing ball results in score for the opponent) is alternated with a reversed pong rules (ball contact is punished, catching ball results in score for the opponent). Task switching schedule can be defined arbitrarily (example in the code).

Environment can be executed in usual display mode or in headless mode without display output (for running)

* Main files:

EnvPongDraft.py: main class for game mechanics; implementation for displaying on screen

EnvPongDraft_Surface_Headless.py: main class for game mechanics; implementation for headless environment without display output

ExpPongDraft.py: Demo of a general game on display, switchting (different modes to switch from classic to reversed pong)

ExpPongTaskSwitch_Draft.py : Demo of different types of switchting behavior (random, alternate)

ExpPongDraft_DisplayVsHeadless.py : Demo testing match in generated data between display and headless pong versions.

* Task description and schedule.

Task 1 : create an openAI gym interface to use the environment in reinforcement learning scenarios
  - creating a class PongSwitch env that contains switchtable pong environment where different pong versions can be changed during training
  - make sure the environment can be also executed in headless mode without display output;
  - Test the environment with swiching behavior with standard RL networks (eg, use predefined stable_baselines algorithms); check execution of the task in headless mode on HPC nodes;

Task 2 : customize the environment to prepare for upcoming continual, multi-task learning experiments
  - parameterize the pong ball behavior: introduce ability to have many balls, each ball having a different shape, different color, different physics (simple variation parameter: speed)
  - parameterize the reward / punishment : make it possible to provide reward after a sequence of actions executed succesfully; 
    * example : to obtain reward, paddle has to hit different balls, say according to their shape (first round, then square - reward; any other sequence - no reward; or punishment for the wrong sequence, etc) or to their color (hit red, blue - reward; otherwise none; some wrong sequences may get punishment etc)
  - introduce a switching behavior in those tasks: example - task that rewards red-blue sequence can switch to a one punishing the very same sequence, etc:
  - introduce a special action so that any neural network that control the environment can ** actively ** trigger task switching. The control of task switching via special actions ("switch task" as an output of action layer of the network) should be possible in different forms. Example : 1. action "Task Switch" triggers transition to a randomly choosen task (network does have a choice to trigger transition, but not to which task); 2. action "Switch to Task ID X" possible, where a task X is selected; 3. probabilistic : network can choose task X - with probability p task X is selected; with probability 1-p transition to a random task Y;

Task 3 : in multi task switchin scenario, implement a network that learns to recognize which environment out of different possible versions is currently played;


* Installation: 

Compatible with Python 2.x and Python 3.x (preferable to work with Python 3.x)
Requirenments:  numpy, parameters, pygame

- prepare virtualenv : 
  
  virtualenv venvs/PongExp
  
  - on HPC nodes, first load modules to install virtualenv:
  ml GCC/8.3.0
  ml Python/3.6.8
  pip install --user virtualenv
  Add path (usually, $HOME/.local/bin) to .bashrc
  now virtualenv can be used as usual)
  
- activate virtual env : 
  
  source venvs/PongExp/bin/activate

  - on HPC nodes, numpy should be loaded ** before ** activating virtual env via module load :
      ml GCC/8.3.0
      ml Python/3.6.8
      ml SciPy-Stack/2019a-Python-3.6.8
      
      source venvs/PongExp/bin/activate
  
  - Install requirenments via pip :  numpy, parameters, pygame
    
    pip install numpy parameters pygame
    
    - on HPC, numpy already loaded via module load:
      
      pip install parameters pygame


* Example runs instructions:

- Testing General Game, Switchting (different modes to switch from classic to reversed pong)
python ExpPongDraft.py

- Testing Switching the Tasks (different modes to switch from classic to reversed pong)
python ExpPongTaskSwitch_Draft.py

- Testing Switching the Tasks and Comparing Display with Headless env version 
python ExpPongDraft_DisplayVsHeadless.py      

