import numpy as np
import numpy.random as rn

class Gridworld(object):

    def __init__(self, size, epsilon, discount):
        
        #we discussed that we'll limit the actions to L,R,U,D 
        #Sourav/Alexa - Should we can modify this to have diagonal moves as well? 
        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        self.n_actions = len(self.actions)
        self.n_states = size**2
        self.size = size
        self.epsilon = epsilon
        self.discount = discount

        #Defining transitions for the gridworld - T(state , action, next state) 
        self.T = np.array(
            [[[self.transitions(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.size, self.epsilon,
                                              self.discount)

    # Each vector consists of n_states elements - with 1 at the state index position 
    def feature_vector(self, i):
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    # We make a feature matrix (n_states x n_states) by calling feature_vector(i) <n_states> times
    def feature_matrix(self):
        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n)
            features.append(f)
        return np.array(features)

    # Converting the index to coordinate
    def state_index_to_coord(self, i):
        return (i % self.size, i // self.size)

    # Converting coordinate to state index
    def coord_to_state_index(self, p):
        return p[0] + p[1]*self.size

    # Check if the states are neighbouring states
    # Action defined is one step in the (L,R,U,D) direction, so states after a step should be neighbouring, unless it's the peripheral state 
    def neighbouring_states(self, i, k):
        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    # Transitions take in the state index and action index and convert them into coordinates
    # 1- Check for neighbouring states 
    # 2- Check if taking the given action actually moves to the specified next state
    def transitions(self, s, a, sp):

        # getting the coordiantes from state, action , next state indices 
        x_s, y_s = self.state_index_to_coord(s)
        x_a, y_a = self.actions[a]
        x_sp, y_sp = self.state_index_to_coord(sp)

        # additional check for states being actually accessible in the gridworld. 
        if not self.neighbouring_states((x_s, y_s), (x_sp, y_sp)):
            return 0.0

        # check if taking the given action actually moves to the specified next state
        if (x_s + x_a, y_s + y_a) == (x_sp, y_sp):
            return 1.0

        # If these are not the same point, then we can move there by epsilon.
        # We might or might not want to include any randomness in our transitions 
        if (x_s, y_s) != (x_sp, y_sp):
            return 0.0

        # Checking for the bounds (within grid size)
        if (x_s, y_s) in {(0, 0), (self.size-1, self.size-1), (0, self.size-1), (self.size-1, 0)}:
            
            #Transition to the corner - if an action results in the transition, we stay at the same place with 1 probability, otherwise 0
            if not (0 <= x_s + x_a < self.size and
                    0 <= y_s + y_a < self.size):
                return 1.0
            else:
                # We can blow off the grid in either direction only by epsilon.
                return 0.0
        else:
            # Not a corner. Is it an edge?
            if (x_s not in {0, self.size-1} and
                y_s not in {0, self.size-1}):
                # Not an edge.
                return 0.0

            # Transition to the edge, checking if an action resulted in the transition
            if not (0 <= x_s + x_a < self.size and 0 <= y_s + y_a < self.size):
                return 1.0
            else:
                return 0.0
    # Need to define multiple rewards for this grid, some closer smaller rewards and some farther higher rewards
    def R(self, state_int):
        if state_int == self.n_states - 1:
            return 1
        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        sx, sy = self.state_index_to_coord(state_int)

        if sx < self.size and sy < self.size:
            return rn.randint(0, 2)
        if sx < self.size-1:
            return 0
        if sy < self.size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        sx, sy = self.state_index_to_coord(state_int)
        if sx < sy:
            return 0
        return 1

    def generate_trajectories(self, n_trajectories, trajectory_length, policy, random_start=False):
        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.size), rn.randint(self.size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                #epsilon-greedy
                if rn.random() < self.epsilon:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    #action comes from the policy (1-epsilon) number of times
                    action = self.actions[policy(self.coord_to_state_index((sx, sy)))]

                #check if the new state produced by taking that action is within the state space range, then move otherwise stay
                if (0 <= sx + action[0] < self.size and
                        0 <= sy + action[1] < self.size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                # elements that make up the tuple for recording trajectories
                state_int = self.coord_to_state_index((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.coord_to_state_index((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)