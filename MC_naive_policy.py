import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from BlackJackEnv import Env

PLOT_DATA = [ ]
def add_plot_data(Q):
    data = np.zeros(shape=(10, 10))
    for i in range(10):
        for j in range(10):
            data[i][j] = Q[i + 1][j + 12]
    PLOT_DATA.append(data)

NEPISODE = 500000
# if you change number of tracked ones, consider change subplots's positions
TRACKED_EPISODES = [ 10000, 500000 ]

# update a state's value
def update(state, Q, N, G):
    u, d, s = state
    N[u][d][s] += 1
    Q[u][d][s] += (G - Q[u][d][s]) / N[u][d][s]

# blackjack environment
env = Env()

# estimate values [usable ace][dealer's showing card (ace/10)][current sum]
Q = np.zeros(shape=(2, 11, 22))
# count number of visits of states
N = np.zeros(shape=(2, 11, 22))

for epi in range(NEPISODE):
    # init environment and state
    env.initEpisode()

    # dealder init cards
    dealer_init_card = [ np.minimum(10, env.hit()), np.minimum(10, env.hit()) ]
    dealer_faceup = dealer_init_card[0]
    dealer_sum = sum(dealer_init_card)
    dealer_has_ace = 1 in dealer_init_card

    # player's init cards
    init_card = [ np.minimum(10, env.hit()), np.minimum(10, env.hit()) ]
    S = sum(init_card)
    has_ace = 1 in init_card

    if has_ace and S == 11: # natural case
        update([1, dealer_faceup, S], Q, N, 1)
        continue

    # generate an episode
    bust = False
    STATES = [ ]
    while True:
        usable_ace = has_ace and S <= 11
        curS = S + (10 if usable_ace else 0)
        STATES.append([ int(usable_ace), dealer_faceup, curS ])

        if curS == 20 or curS == 21:
            break

        draw = np.minimum(10, env.hit())
        S += draw
        if draw == 1:
            has_ace = True

        if S > 21: # go bust
            bust = True
            break

    # dealer's turn
    R = 0
    if bust:
        R = -1
    else:
        if has_ace and S <= 11:
            S += 10

        while True:
            dealer_cur_sum = dealer_sum + (10 if (dealer_has_ace and dealer_sum <= 11) else 0)

            if dealer_cur_sum >= 17:
                break

            draw = np.minimum(10, env.hit())
            dealer_sum += draw
            if draw == 1:
                dealer_has_ace = True

        if dealer_has_ace and dealer_sum <= 11:
            dealer_sum += 10

        if dealer_sum > 21 or dealer_sum < S:
            R = 1
        elif dealer_sum > S:
            R = -1
        else:
            R = 0

    for state in reversed(STATES):
        update(state, Q, N, R)

    if epi + 1 in TRACKED_EPISODES:
        add_plot_data(Q[0])
        add_plot_data(Q[1])

        print('added')

# plot
fig, axis = plt.subplots(nrows=2, ncols=2)

counter = 0
for ax in np.reshape(axis, newshape=2*2):
    im = ax.imshow(PLOT_DATA[counter], vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    counter += 1

axis[1][1].set_xticks([0, 9], ['12', '21'])
axis[1][1].set_yticks([0, 9], ['A', '10'])
axis[1][1].set_xlabel('Tổng điểm')
axis[1][1].set_ylabel('Bài ngửa của nhà cái')

axis[0][0].set_title('Coi quân Át 1 điểm', size='large')
axis[0][1].set_title('Coi quân Át 11 điểm', size='large')
axis[0][0].set_ylabel('Sau {} episode'.format(TRACKED_EPISODES[0]), size='large')
axis[1][0].set_ylabel('Sau {} episodes'.format(TRACKED_EPISODES[1]), size='large')

fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)

fig.tight_layout()
plt.show()

