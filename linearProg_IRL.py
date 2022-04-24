import numpy as np
import random
from cvxopt import matrix, solvers

def v_tensor(v_pi, transition, d_states, n_states, n_actions, pi):
    v = np.zeros((n_states, n_actions-1 , d_states))
    for i in n_states:
        a = pi[i]
        


def generate_G_matrix(n_states, n_actions, d_states, v_vector, x_dim):

    bottom_row = np.vstack([
                    np.hstack([
                        np.ones((n_actions-1, 1)).dot(np.eye(1, n_states, l)),
                        np.hstack([-np.eye(n_actions-1) if i == l
                                   else np.zeros((n_actions-1, n_actions-1))
                         for i in range(n_states)]),
                        np.hstack([2*np.eye(n_actions-1) if i == l
                                   else np.zeros((n_actions-1, n_actions-1))
                         for i in range(n_states)]),
                        np.zeros((n_actions-1, d_states))])
                    for l in range(n_states)])
    assert bottom_row.shape[1] == x_dim
    G = np.vstack([
            np.hstack([
                np.zeros((d_states, n_states)),
                np.zeros((d_states, n_states*(n_actions-1))),
                np.zeros((d_states, n_states*(n_actions-1))),
                np.eye(d_states)]),
            np.hstack([
                np.zeros((d_states, n_states)),
                np.zeros((d_states, n_states*(n_actions-1))),
                np.zeros((d_states, n_states*(n_actions-1))),
                -np.eye(d_states)]),
            np.hstack([
                np.zeros((n_states*(n_actions-1), n_states)),
                -np.eye(n_states*(n_actions-1)),
                np.zeros((n_states*(n_actions-1), n_states*(n_actions-1))),
                np.zeros((n_states*(n_actions-1), D))]),
            np.hstack([
                np.zeros((n_states*(n_actions-1), n_states)),
                np.zeros((n_states*(n_actions-1), n_states*(n_actions-1))),
                -np.eye(n_states*(n_actions-1)),
                np.zeros((n_states*(n_actions-1), D))]),
            bottom_row])
    assert G.shape[1] == x_dim
    return G, bottom_row

def linear_irl(v_pi, transition, feature_mat, n_states, n_actions, pi):
    
    print(feature_mat.shape)
    #feature matrix is of the shape (n_states * n_states) - d_states gets feature dimensionality of each state 
    d_states = feature_mat.shape[1]
    
    # helper tensor 
    v_vector = v_tensor(v_pi, transition, d_states, n_states, n_actions, pi)

    x_dim = n_states + ((n_actions - 1) * n_states ) * 2 + d_states

    c = -np.hstack([np.ones(n_states), np.zeros(((n_actions - 1) * n_states ) * 2 + d_states)])
    #check for the shape

    helper_A = np.vstack([v_vector[i,j].T for i in range(n_states) for j in range(n_actions-1)])

    A = np.hstack([ np.zeros(n_states * (n_actions-1) , n_states),
    np.eye(n_states * (n_actions-1)),
    -np.eye(n_states * (n_actions-1)),
    helper_A])
    # check for the shape

    B = np.zeros(A.shape[0])

    G , bottom_row = generate_G_matrix(n_states, n_actions, d_states, v_vector, x_dim)

    h = np.vstack([np.ones((d_states*2, 1)),
                   np.zeros((n_states*(n_actions-1)*2+bottom_row.shape[0], 1))])


    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(B)
    results = solvers.lp(c, G, h, A, b)
    alpha = np.asarray(results["x"][-d_states:], dtype=np.double)
    return np.dot(feature_mat, -alpha)

