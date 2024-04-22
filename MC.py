from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from BlackJackEnv import Env

NEPISODE = 500000
TRACKED_EPISODES = [ 10000, 500000 ]

EPSILON = 0.1

env = Env()

def chooseAction(state):
    if np.random.sample() < EPSILON:
        print('hit')
    else:
        print('not hit')

for epi in range(NEPISODE):
    # init episode
    env.initEpisode()

    dealer_card = env.hit() # face up
    env.hit() # dealer's face down card

    if (epi + 1) in TRACKED_EPISODES:
        print()

