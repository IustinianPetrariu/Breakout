import gym
import time
from PreprocessAtari import PreprocessAtari
from FrameBuffer import FrameBuffer
import matplotlib.pyplot as plt

# 0  'NOOP'
# 1  'FIRE'
# 2  'RIGHT'
# 3  'LEFT' 

def make_env():
    env = gym.make("BreakoutDeterministic-v4")
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4)
    return env

env = make_env()
env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape


for _ in range(50):
    obs, _, _, _ = env.step(env.action_space.sample())


plt.title("Game image")
plt.imshow(env.render("rgb_array"))
plt.show()

plt.title("Agent observation (4 frames left to right)")
plt.imshow(obs.transpose([0,2,1]).reshape([state_dim[0],-1]))
plt.show()

# env = gym.make("BreakoutNoFrameskip-v4")
# env = PreprocessAtari(env)

# print("Observation Space: ", env.observation_space)
# print("Action Space       ", env.action_space)
# print("am afisat")
# print(env.observation_space.high.shape)
# print(env.observation_space.low.shape)


# obs = env.reset()
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(200):
#         env.render()
#         time.sleep(0.05)
#         # print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()