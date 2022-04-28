from itertools import product

import numpy as np
import numpy.random as rn

import ValueIteration

def irl(feature_mat, n_actions, discount, transition_probability, trajectories, iterations, learning_rate):
    
    #number of states and dimensionaloity of the feature vector
    n_states, d_states = feature_mat.shape

    # Initialise weights.
    weights = rn.uniform(size=(d_states,))

    # Calculate the feature expectations from obtained trajectories
    feature_expectations = calc_feature_expectations(feature_mat, trajectories, n_states, d_states)

    # Gradient descent is defined here
    #
    for i in range(iterations):
        r = feature_mat.dot(weights)
        state_visitation_freq = state_visitations(n_states, r, n_actions, discount, transition_probability, trajectories)
        grad = feature_expectations - feature_mat.T.dot(state_visitation_freq)

        weights += learning_rate * grad

    return feature_mat.dot(weights).reshape((n_states,))

def find_svf(n_states, trajectories):

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def calc_feature_expectations(feature_matrix, trajectories, n_states, d_states):
    feature_exp = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, _ , _ in trajectory:
            feature_exp += feature_matrix[int(state)]

    feature_exp /= trajectories.shape[0]

    return feature_exp

def state_visitations(n_states, r, n_actions, discount, transition_probability, trajectories):

    n_trajectories = trajectories.shape[0]
    trajectory_length = n_trajectories * 3

    v, policy = ValueIteration.value_iteration(n_states, n_actions, transition_probability, r)

    start_s = np.zeros(n_states)
    for trajectory in trajectories:
        idx = trajectory[0][0]
        start_s[int(idx)] += 1
    prob_start_s = start_s/n_trajectories

    state_visitation = np.tile(prob_start_s, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        state_visitation[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            state_visitation[k, t] += (state_visitation[i, t-1] * policy[i] * transition_probability[i, j, k])

    return state_visitation.sum(axis=1)