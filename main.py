import numpy as np
import tensorflow as tf
import gym
import warnings
from gym import wrappers
from Model import Model
from Agent import Agent 

tf.compat.v1.disable_eager_execution()
warnings.filterwarnings("ignore")


if __name__ == '__main__':
	env = gym.make("Breakout-ram-v0")

	actions_number = env.action_space.n   # 4
	state_number   = env.reset().shape[0] # (128,) 

	actor  = Model(state_number, actions_number)
	critic = Model(state_number, actions_number)

	agent = Agent(actor, critic,  env, num_frames=int(8e6)) # train_nums=int(7e4)
	
	agent.train("Actor/dqn_model")
   











