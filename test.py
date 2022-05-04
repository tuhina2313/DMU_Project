import gridworld
import ValueIteration
#import linearProg_IRL
import maxEntIRL
#import deepIRL
#import linear_irl

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

def generate_plots(expert_reward, irl_reward, size):
    x_label = np.arange(size * size)
    plt.plot(list(x_label), list(expert_reward), color = "b", label = "Expert Reward")
    plt.plot(list(x_label), list(irl_reward), color = "g", label = "IRL Reward")

    plt.xlabel("State Index")
    plt.ylabel("Reward R(s)")

    plt.show()

if __name__ == '__main__':
    size = 5
    n_episodes = 200
    n_trajectories = 20
    epsilon = 0.3
    discount = 0.95
    epochs = 200
    learning_rate = 0.01

    gw = gridworld.Gridworld(size, epsilon, discount)

    v, optimal_policy = test_for_gw(gw, size, n_episodes, epsilon, discount)
    print("Value function", v)
    print("Policy", optimal_policy)

    traj_length = 3 * size
    trajectories = test_for_traj(gw, n_trajectories , traj_length, optimal_policy)
    print("Generated Trajectories")

    print(trajectories)
    x_traj = np.arange(0, len(trajectories), 20)
    y_traj = [len(trajectories[i]) for i in x_traj]
    sum_traj = [len(traj) for traj in trajectories]
    avg_traj = sum(sum_traj) / len(trajectories)
    print(avg_traj)
    plt.plot(list(x_traj), list(y_traj), color = "b", label = "Length of Trajectories")
    plt.title ("Average length of Trajectories = ")

    feature_mat = gw.feature_matrix()
    print("Created feature matrix")
    expert_reward = np.array([gw.R(s) for s in range(gw.n_states)])
    irl_reward = maxEntIRL.irl(feature_mat, gw.n_actions, discount, gw.T, trajectories, epochs, learning_rate)
    #irl_reward =linear_irl.irl(gw.n_states, gw.n_actions, gw.T, optimal_policy, discount, 10, 5)
    # irl_reward = deepIRL.irl((5,5), feature_mat, gw.n_actions, discount, gw.T,
    #     trajectories, epochs, learning_rate, initialisation="normal", l1=0.1,
    #     l2=0.1)

    generate_plots(expert_reward, irl_reward, size)

    plt.subplot(1, 2, 1)
    plt.pcolor(expert_reward.reshape((size, size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(irl_reward.reshape((size, size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

    # structure = (5, 5)
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









