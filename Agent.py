import tensorflow as tf
import numpy as np 
import time
from tensorflow.keras import optimizers

class Agent:

	def __init__(self, actor, critic, env, num_frames=10000):

		self.actor = actor
		self.critic = critic
		self.actor.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
													loss='mse',
													metrics=['accuracy'])
		self.env = env
		self.num_frames     	= num_frames
		self.learning_rate 		= 0.0015
		self.gamma 				= 0.95
		self.epsilon 			= 0.6
		self.critic_update  	= 40000
		self.start_learning 	= 1000
		self.batch_size     	= 32
		self.save_every_n_step 	= 5000
		self.buffer_size    	= 10000
		self.epsilon_decay  	= 0.99
		self.learn_every_n_step = 32
		self.num_in_buffer  	= 0
		self.min_epsilon    	= 0.1

		self.obs 			= np.empty((self.buffer_size,) + self.env.reset().shape)
		self.actions 		= np.empty((self.buffer_size), dtype = np.int8)
		self.rewards 		= np.empty((self.buffer_size), dtype = np.float32)
		self.dones 			= np.empty((self.buffer_size), dtype = np.bool)
		self.next_states 	= np.empty((self.buffer_size, ) + self.env.reset().shape)
		self.next_idx 		= 0


	def train(self, model_path):
		episode = 0
		step = 0
		
		while step < self.num_frames: 

			obs = self.env.reset()
			#normalize observation 
			obs = obs / 256

			done = False
			episode_reward = 0.0

			while not done: 
				# self.env.render()
				# time.sleep(0.05)
				step += 1
				best_action, q_values = self.actor.take_action(obs[None])
				action = self.get_action(best_action)

				self.epsilon = max(self.epsilon, self.min_epsilon)
				next_obs, reward, done, info = self.env.step(action)

				#normalize next observation 
				next_obs = next_obs / 256

				episode_reward += reward
				
				self.experience_replay(obs, action, reward, next_obs, done)
				obs = next_obs

				self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)
		
				if step > self.start_learning:

					if not step % self.learn_every_n_step:
						self.train_step()
						
					if step % self.save_every_n_step == 0:
						self.save_model(model_path)

					if step % self.critic_update == 0:
						self.update_critic()
			
			if step > self.start_learning:
				self.epsilon = self.epsilon * self.epsilon_decay

			print("episode: ", episode, ' step: ', step,  '-> reward: ', episode_reward)
			episode += 1


	def train_step(self):
		idxes 		= self.sample()	
		s_batch 	= self.obs[idxes]  # shape = (batch_size=32, 128) obs.shape[0] = 128
		a_batch 	= self.actions[idxes]
		r_batch 	= self.rewards[idxes]
		ns_batch 	= self.next_states[idxes]
		done_batch 	= self.dones[idxes]

		target_q = r_batch + self.gamma * np.amax(self.get_critic_choice(ns_batch), axis = 1)  * (1-done_batch)
		target_f = self.actor.predict(s_batch) # shape = (32,4)

		for i, val in enumerate(a_batch):
			target_f[i][val] = target_q[i]

		self.actor.train_on_batch(s_batch, target_f)


	def experience_replay(self, obs, action, reward, next_state, done):
		n_idx = self.next_idx % self.buffer_size
		self.obs[n_idx] = obs
		self.actions[n_idx] = action
		self.rewards[n_idx] = reward
		self.next_states[n_idx] =  next_state
		self.dones[n_idx] =  done 
		self.next_idx  = (self.next_idx + 1) % self.buffer_size


	# take 32 differet action 
	def sample(self):
		return np.random.choice(self.num_in_buffer, self.batch_size, replace = False)
	

    # e-greedy pass
	def get_action(self, best_action):
		if np.random.rand() < self.epsilon:
			#take a random action in game
			action = self.env.action_space.sample()
		else:
			#take an action according to model  
			action = best_action
		return action


	def update_critic(self):
		self.critic.set_weights(self.actor.get_weights())


	def get_critic_choice(self, obs):
		return self.critic.predict(obs)


	def save_model(self, model_path):
		self.actor.save_weights(model_path)




