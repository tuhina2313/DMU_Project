import numpy as np
import gridworld as gw

def value_iteration(m, discount):
    #Takes in a discount and grid world m with defined Ts, Rs
    #Outputs V(s) vector 
    
    nactions = m.n_actions
    nstates= m.n_states
    eps = 0.001
    V = np.zeros(nactions)
    Vmax = np.zeros(nstates)
    oldV = [10000]*nstates
    difference = float("inf")
    optimal_policy = np.zeros(nstates)

    while difference > eps:
        difference = 0
        for s_index in range(nstates):
            reward = m.R(s_index) #here i is the state index
            
            for a_index in range(nactions):
                tsum=0

                for sp_index in range(nstates):
                    tsum = tsum + m.T[s_index,a_index,sp_index] * Vmax[sp_index]
                    
                V[a_index] = reward+discount*tsum 
            
            new_difference = abs(Vmax[s_index] - max(V))
            if new_difference > difference:
                difference = new_difference    
            Vmax[s_index]=max(V)
            optimal_policy[s_index] = np.argmax(V)
        oldV = Vmax
    return Vmax, optimal_policy

    #Questions: 
    #Do we have actions(m) and states(m)? Are these lists? 
    
#Homework Solution-- don't fully understand but we can use if we alter
    #while max(abs(V-oldV)) > 0.0:
     #   oldV = V
      #  V = max(Rs[a] + discount*Ts[a]*V for a in keys(Rs))
   # return V 