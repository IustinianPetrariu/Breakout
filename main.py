import numpy as np
import tensorflow as tf
import gym
import os
import datetime
import warnings
from gym import wrappers
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings("ignore")
import tensorflow.keras.optimizers as ko
import time
from Model import Model
from Agent import Agent 


np.random.seed(1)
tf.random.set_seed(1)


def test_model():

    env = gym.make('Breakout-ram-v0')
    print('num_actions: ', env.action_space.n)

    model = Model(128, env.action_space.n)

    obs = env.reset()
    print('obs_shape: ', obs.shape)

    # tensorflow 2.0: no feed_dict or tf.Session() needed at all
    best_action, q_values = model.action_value(obs)
    print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]


if __name__ == '__main__':

	test_model()
	print('nimic')
	env = gym.make("Breakout-ram-v0")
	# env = wrappers.Monitor(env, os.path.join(os.getcwd(), 'video_breakout'), force = True)
	# env = wrappers.RecordVideo(env, os.path.join(os.getcwd(), 'video_breakout'), force = True)
	num_actions = env.action_space.n # 4
    
	num_state = env.reset().shape[0] # (128,) 

	model = Model(num_state, num_actions)

	# model = model.get_network() 
	target_model = Model(num_state, num_actions)
	# target_model = target_model.get_network()
	agent = Agent(model, target_model,  env, train_nums=int(1e6)) # train_nums=int(7e4)
	
	agent.train("new_dqn/dqn_checkpoint")

	print("train is over and model is saved")

	np.save('dqn_agent_train_lost.npy', agent.loss_stat)
   













