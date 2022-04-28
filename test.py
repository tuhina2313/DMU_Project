import gridworld
import ValueIteration
#import linearProg_IRL
import maxEntIRL
#import deepIRL

import sys
import numpy as np
import matplotlib.pyplot as plt

def test_for_gw(gw, size, n_episodes, epsilon, discount):
    feature_mat = gw.feature_matrix()

    reward = np.array([gw.R(i) for i in range(gw.n_states)])
    v, optimal_policy = ValueIteration.value_iteration(gw.n_states, gw.n_actions, gw.T, reward)

    return v, optimal_policy

def test_for_traj(gw, n_episodes, traj_length, optimal_policy):
    trajectories = gw.generate_trajectories(n_episodes, traj_length, optimal_policy)
    #print(trajectories)
    return trajectories

if __name__ == '__main__':
    size = 5
    n_episodes = 200
    epsilon = 0.3
    discount = 0.9

    gw = gridworld.Gridworld(size, epsilon, discount)

    v, optimal_policy = test_for_gw(gw, size, n_episodes, epsilon, discount)
    print("Value function", v)
    print("Policy", optimal_policy)

    traj_length = 4 * size
    trajectories = test_for_traj(gw, 200 , traj_length, optimal_policy)

    feature_mat = gw.feature_matrix()
    expert_reward = np.array([gw.R(s) for s in range(gw.n_states)])
    irl_reward = maxEntIRL.irl(feature_mat, gw.n_actions, discount, gw.T, trajectories, 200, 0.01)

    plt.subplot(1, 2, 1)
    plt.pcolor(expert_reward.reshape((size, size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(irl_reward.reshape((size, size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

    # structure = (3, 3)
    # epochs = 50
    # learning_rate = 0.01
    # l1 = l2 = 0
    # r = deepIRL.irl((feature_mat.shape[1],) + structure, feature_mat, gw.n_actions, discount, gw.transition_probability, trajectories, epochs, learning_rate, l1=l1, l2=l2)
    # plt.subplot(1, 2, 1)
    # plt.pcolor(expert_reward.reshape((size, size)))
    # plt.colorbar()
    # plt.title("Groundtruth reward")
    # plt.subplot(1, 2, 2)
    # plt.pcolor(r.reshape((size, size)))
    # plt.colorbar()
    # plt.title("Recovered reward")
    # plt.show()








