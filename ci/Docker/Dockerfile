FROM python:3.6
MAINTAINER Josef Azzam <j.azzam@fz-juelich.de>

# pipeline tools
RUN pip3 install prospector bandit mypy pytest anybadge coverage

# standard libraries
RUN pip3 install numpy scipy matplotlib jupyter gcovr

# deeprl libs
RUN pip3 install psutil opencv-python==4.2.0.34 gym gym[atari] stable_baselines parameters pygame tensorflow==1.14 ray[rllib] ipykernel pyzmq
