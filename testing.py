import gym
import time
from gym.utils import play


# gameplay
# env = gym.make('Breakout-v0')
# play.play(env, zoom=3)


env = gym.make('Breakout-ram-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        time.sleep(1)
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()