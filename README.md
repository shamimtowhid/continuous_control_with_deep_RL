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
