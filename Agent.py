import tensorflow as tf 
import numpy as np 
import time

class Agent:

	def __init__(self, model, target_model, env, buffer_size=10000, learning_rate=.0015, epsilon=0.6, epsilon_dacay=0.999,
                 min_epsilon=.1, gamma=.95, batch_size=32, target_update_iter=1000, learn_every_n_step=32, train_nums=10000, 
                 start_learning=100, save_every_n_step = 5000):

		self.model = model
		self.target_model = target_model
		self.opt = tf.keras.optimizers.RMSprop(learning_rate = learning_rate, clipvalue = 1.0) #, clipvalue = 10.0
		self.model.compile(optimizer = self.opt, loss = 'huber_loss')

		self.env = env
		self.lr = learning_rate
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_dacay
		self.min_epsilon = min_epsilon
		self.gamma = gamma
		self.batch_size = batch_size
		self.target_update_iter = target_update_iter
		self.train_nums = train_nums
		self.num_in_buffer = 0
		self.buffer_size = buffer_size
		self.start_learning = start_learning
		self.learn_every_n_step = learn_every_n_step
		self.save_every_n_step = save_every_n_step

		self.obs = np.empty((self.buffer_size,)+ self.env.reset().shape)
		self.actions = np.empty((self.buffer_size), dtype = np.int8)
		self.rewards = np.empty((self.buffer_size), dtype = np.float32)
		self.dones = np.empty((self.buffer_size), dtype = np.bool)
		self.next_states = np.empty((self.buffer_size, ) + self.env.reset().shape)

		self.next_idx = 0
		self.loss_stat = []
		self.reward_his = []


	def train(self, model_path_dir):

		episode = 0
		step = 0
		loss = 0
		
		while step < self.train_nums: # 70000

			obs = self.env.reset()
			obs = normalize_obs(obs)

			done = False
			episode_reward = 0.0

			while not done: 
				self.env.render()
				# time.sleep(0.05)
				step += 1
				best_action, q_values = self.model.action_value(obs[None])
				action = self.get_action(best_action)

				self.epsilon = max(self.epsilon, self.min_epsilon)

				next_obs, reward, done, info = self.env.step(action)
				next_obs = normalize_obs(next_obs)

				episode_reward += reward
				
				self.store_transition(obs, action, reward, next_obs, done)
				obs = next_obs

				self.num_in_buffer = min(self.num_in_buffer+1, self.buffer_size)
		
				if step > self.start_learning:
					if not step % self.learn_every_n_step:
						# print(" -- step : ", step, ' -- mod: ', step % self.learn_every_n_step)
						losses = self.train_step()
						self.loss_stat.append(losses)
					if step % self.save_every_n_step == 0:
						print(' losses each {} steps: {}'.format(step, losses))
						self.save_model(model_path_dir)

					if step % self.target_update_iter == 0:
						self.update_target_model()
			
			if step > self.start_learning:
				self.e_decay()

			print("--episode: ", episode, '-- step: ', step,  '--reward: ', episode_reward)
			episode += 1

			self.reward_his.append(episode_reward)


	def train_step(self):
		idxes = self.sample(self.batch_size)	
		s_batch = self.obs[idxes]  # shape = (batch_size=32, 128) obs.shape[0] = 128
		a_batch = self.actions[idxes]
		r_batch = self.rewards[idxes]
		ns_batch = self.next_states[idxes]
		done_batch = self.dones[idxes]

		target_q = r_batch + self.gamma * np.amax(self.get_target_value(ns_batch), axis = 1)*(1-done_batch)
		target_f = self.model.predict(s_batch) # shape = (32,4)

		for i, val in enumerate(a_batch):
			target_f[i][val] = target_q[i]

		losses = self.model.train_on_batch(s_batch, target_f)
		return losses
	

	def evaluation(self, env, render = False):
		obs, done, ep_reward = env.reset(), False, 0
		while not done:
			action, q_values = self.model.action_value(obs[None])
			obs, reward, done, info = env.step(action)
			ep_reward += reward
			if render:
				env.render()
			time.sleep(0.05)
		env.close()
		return ep_reward


	def store_transition(self, obs, action, reward, next_state, done):

		n_idx = self.next_idx % self.buffer_size
		self.obs[n_idx] = obs
		self.actions[n_idx] = action
		self.rewards[n_idx] = reward
		self.next_states[n_idx] =  next_state
		self.dones[n_idx] =  done 
		self.next_idx  = (self.next_idx + 1) % self.buffer_size


	# sample n different indexes
	def sample(self, n):
		assert n<self.num_in_buffer
		return np.random.choice(self.num_in_buffer, self.batch_size, replace = False)
	
    # e-greedy
	def get_action(self, best_action):
		if np.random.rand() < self.epsilon:
			action = self.env.action_space.sample()
		else:
			action = best_action
		return action

    # assign the current network parameters to target network
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def get_target_value(self, obs):
		return self.target_model.predict(obs)

	def e_decay(self):
		self.epsilon = self.epsilon * self.epsilon_decay

	def save_model(self, model_path_dir):

		# tf.keras.models.save_model(self.model, model_path_dir)
		# tf.saved_model.save(self.model, model_path_dir)
		self.model.save_weights(model_path_dir)


def normalize_obs(obs, scale = 256):
	return obs/scale






