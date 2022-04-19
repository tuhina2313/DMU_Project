import gridworld
import ValueIteration
#import linearProg_IRL

import sys
import numpy as np
import matplotlib.pyplot as plt

def test_for_gw(size, n_episodes, epsilon, discount):
    n_states = len(gridworld.states)
    n_actions = len(gridworld.actions)
    gw = gridworld.Gridworld(size, epsilon, discount)
    feature_mat = gridworld.feature_matrix()

    reward = np.array([gridworld.R(i) for i in n_states])
    optimal_policy = ValueIteration.optimal_policy()

if __name__ == '__main__':
    gw = gridworld.Gridworld(3, 0.3, 0.9)
    v, optimal_policy = ValueIteration.value_iteration(gw, 0.9)

    print("Value function",v)
    print("Policy", optimal_policy)


