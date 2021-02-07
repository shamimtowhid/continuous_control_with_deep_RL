# Continuous Control with Deep RL

This repo contains implementation of a very popular algorithm used in deep Reinforcement Learning named DDPG (Deep Deterministic Policy Gradient). This project is done as a
part of Udacity's Deep Reinforcement Learning [nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program. 

In this project, I tried to solve a Unity ML agent environment named [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher). 
There are two versions of this environment.
  - Environment 1: contains a single agent.
  - Environment 2: contains 20 identical agents, each with its own copy of the environment.

The second version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of 
gathering experience.
To pass the project I needed to solve any one of the environment. In this repo I solved the first environment. 

## Environment 
![The environment image will be shown here](images/reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

## Set up the environment
Please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in *README.md* at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

You do not need to install Unity for this project. The environment can be downloaded from the following links. It is provided by the Udacity Nanodegree program. 

  - Linux [clicl here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
  - Mac OSX [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
  - Windows (32 bit) [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
  - Windows (64 bit) [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
  
 After Downloading the unity environment, please change the environment path in *Continuous_Control.ipynb* according to your settings.

## How to navigate to this repository
To follow along, please look into the *Contrinuous_Control.ipynb*. In this notebook the agent is trained using DDPG algorithm. The agent is defined in *agent.py* and the Neural Network definition is in *model_definition.py*. I used the Udacity's workspace to train the model. To setup the local environment please follow the guideline provided by [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).  

