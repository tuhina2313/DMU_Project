import gym
import sys
import os
import time
import copy
import random
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt

# define colors
# 0: black (wall); 1 : gray (empty); 2 : blue (agent); 3 : green (pos reward terminal state); 4 : red (neg reward terminal state), 5: purple (small negative reward)
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], \
          2:[0.0,0.0,1.0], 3:[0.0,1.0,0.0], \
          4:[1.0,0.0,0.0], 5:[1.0,0.0,1.0], \
          6:[1.0,1.0,0.0]}

class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_env = 0 
    def __init__(self, map_file='plan0.txt', transition_noise=0):
        self._seed = 0
        self.actions = [0, 1, 2, 3, 4] # Stay, Left, Right, Down, Up
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}
        self.finished = False

        ''' set observation space '''
        self.obs_shape = [128, 128, 3]  # observation space shape
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)
    
        ''' initialize system state ''' 
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, map_file)        
        self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map

        ''' agent state: start, target, current state '''
        self.agent_start_state = self._get_agent_start_state(self.start_grid_map)
        self.start_grid_map[self.agent_start_state] = 1
        self.agent_state = copy.deepcopy(self.agent_start_state)


        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self.grid_map_shape = self.start_grid_map.shape

        self.num_states = np.prod(self.start_grid_map.shape)
        self.num_actions = len(self.actions)
        self._transition_noise = transition_noise


        ''' set other parameters '''
        self.restart_once_done = False  # restart or not once done
        self.verbose = False # to show the environment or not
        self.render_init = False
        GridworldEnv.num_env += 1

    def get_state_ranges(self):
        '''
        Returns range of valid state values, exclusive.
        '''
        return np.array([ [0, self.start_grid_map.shape[0]], [0, self.start_grid_map.shape[1]]])

    def R(self, state, action, next_state):
        if action == 0: return 0
        if self.start_grid_map[next_state[0], next_state[1]] == 3:
            return 10
        elif self.start_grid_map[next_state[0], next_state[1]] == 4:
            return -10
        elif self.start_grid_map[next_state[0], next_state[1]] == 5:
            return -1
        return 0

    def can_move(self, state, action, next_state):
        # Returns True if next_state is valid terrain to move into and not at a terminal state
        # Returns False otherwise
        if next_state[0] < 0 or next_state[0] >= self.grid_map_shape[0]: return False
        if next_state[1] < 0 or next_state[1] >= self.grid_map_shape[1]: return False
        if self.start_grid_map[next_state[0], next_state[1]] == 0: return False # Can't move to walls
        if self.start_grid_map[state[0], state[1]] == 0: return False # Can't move from walls
        if self.start_grid_map[state[0], state[1]] == 3: return False # Can't move from terminal states
        if self.start_grid_map[state[0], state[1]] == 4: return False # Can't move from terminal states
        return True

    def is_terminal(self,state):
        if self.start_grid_map[state[0], state[1]] == 3 or self.start_grid_map[state[0], state[1]] == 4: return True
        return False

    def T(self, state, action, next_state=None):
        possible_next_states = [] # (Prob., State) tuples

        # Add probability of each non-zero likelihood state
        # Self-transitions for invalid actions
        target_next_state = (state[0] + self.action_pos_dict[action][0],
                             state[1] + self.action_pos_dict[action][1])
        
        if self.can_move(state, action, target_next_state):
            possible_next_states = [(1.- self._transition_noise, target_next_state)] 
        else: 
            possible_next_states = [(1.- self._transition_noise, state)] 

        if self._transition_noise > 0:
            other_actions = [0] # No-OP as default case
            if action == 1 or action == 2: 
                other_actions = [3,4]
            elif action == 3 or action == 4:
                other_actions = [1,2]          

            for a in other_actions:
                pos_next_state = (state[0] + self.action_pos_dict[a][0],
                                  state[1] + self.action_pos_dict[a][1])
                if self.can_move(state, a, pos_next_state):
                    possible_next_states.append((self._transition_noise / len(other_actions), 
                                                (state[0] + self.action_pos_dict[a][0],
                                                state[1] + self.action_pos_dict[a][1])))
                else:
                    possible_next_states.append((self._transition_noise / len(other_actions), 
                                                (state[0], state[1])))
        if next_state is not None:
            # Tally up probability for next_state
            probability_mass_next_state = 0
            for outcome in possible_next_states:
                if False not in [outcome[1][i] == next_state[i] for i in range(len(next_state))]:
                    probability_mass_next_state += outcome[0]
            return [(probability_mass_next_state, next_state)]

        return possible_next_states

    def step(self, action):
        ''' return next observation, reward, finished, success '''
        if self.finished is True: return self.agent_state, 0, True, {'success':True}
        action = int(action)
        info = {}
        info['success'] = True        

        # Reset previous agent state position to original color
        #self.current_grid_map[self.agent_state] = self.start_grid_map[self.agent_state]

        cur_state = copy.deepcopy(self.agent_state)

        # Sample from T(s,a) for next state
        distribution = []
        total_prb = 0
        state_dist = self.T(self.agent_state, action)
        for entry in state_dist:
            prb = entry[0]
            state = entry[1]
            if prb > 0:
                total_prb += prb
                distribution.append((total_prb, state))
        random_number = random.random() * total_prb              
        next_state = None
        for sample in distribution:
            if random_number < sample[0]:
                next_state = sample[1]
                break
        if next_state is None:
            info['success'] = False
            raise Exception("No valid next state in transition function")

        # Set new agent state position to 'blue' (2)
        # self.current_grid_map[next_state] = 2
        self.agent_state = copy.copy(next_state)
        # Update scene rendering
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        reward = self.R(cur_state, action, next_state)
        self.finished = self.is_terminal(next_state)

        return self.agent_state, reward, self.finished, info

    def reset(self):
        self.finished = False
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        return self.agent_state

    def _read_grid_map(self, grid_map_path):
        with open(grid_map_path, 'r') as f:
            grid_map = f.readlines()
        grid_map_array = np.zeros([len(grid_map), len(grid_map[0].split())])

        for lnum, line in enumerate(grid_map):
            values = np.array([int(i) for i in line.split()])
            grid_map_array[lnum,:] = values

        return grid_map_array

    def _get_agent_start_state(self, start_grid_map):
        start_state = None
        target_state = None
        for i in range(len(start_grid_map)):
            for j in range(len(start_grid_map[i])):
                if start_grid_map[(i,j)] == 2:
                    start_state = (i,j)
                    return start_state

        if start_state == None:
            raise Exception('Start or target state not specified')
        return start_state

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        grid_map = copy.deepcopy(grid_map)
        grid_map[self.agent_state] = 2
        observation = np.zeros(obs_shape, dtype=np.float32)
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                observation[i*gs0:(i+1)*gs0, j*gs1:(j+1)*gs1] = np.array(COLORS[grid_map[i,j]])
        return observation
  
    def render(self):
        if self.render_init is False:
            self.this_fig_num = GridworldEnv.num_env 
            self.fig = plt.figure(self.this_fig_num)
            plt.show(block=False)
            plt.axis('off')

        img = self.observation
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)
        return 

    def change_start_state(self, sp):
        ''' change agent start state '''
        ''' Input: sp: new start state '''
        if self.agent_start_state[0] == sp[0] and self.agent_start_state[1] == sp[1]:
            _ = self.reset()
            return True
        elif self.start_grid_map[sp[0], sp[1]] == 0:
            return False
        else:
            s_pos = copy.deepcopy(self.agent_start_state)
            self.start_grid_map[s_pos[0], s_pos[1]] = 1
            self.start_grid_map[sp[0], sp[1]] = 2
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_start_state = (sp[0], sp[1])
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self.reset()
        return True
          
    def get_agent_state(self):
        ''' get current agent state '''
        return self.agent_state

    def get_start_state(self):
        ''' get current start state '''
        return self.agent_start_state

    def _close_env(self):
        plt.close(1)
        return
    