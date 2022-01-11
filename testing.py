# import gym
# import time
# from gym.utils import play

import matplotlib.pyplot as plt
import pandas as pd


# gameplay
# env = gym.make('Breakout-v0')
# play.play(env, zoom=3)

df = pd.read_csv('csv_results.csv')
# print(df.head())
ax = df.plot.hist()
ax.figure.savefig('histogram.pdf')


ax = df.plot()
ax.figure.savefig('graph.pdf')

# df.set_index('Results').plot()
# plt.show()

# ax = df.hist()  # s is an instance of Series
# fig = ax.get_figure()
# fig.savefig('./figure.pdf')