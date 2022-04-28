

class RLAlgorithms:
    def __init__(self, rewards, transitions, states, actions, gamma):
        self.rewards = rewards
        self.transitions = transitions
        self.states = states
        self.actions = actions
        self.gamma = gamma

    def populate_value_table(self, table, val):
        for s in self.states:
            table[s] = val
        return table

    def update_value(self, state, value_table):
        best_val = -float('inf')
        for a in self.actions:
            avg_val = 0.0
            for s in self.states:
                avg_val += self.transitions[(state, a, s)] \
                           * (self.rewards[(state, a, s)] + self.gamma * value_table[s])
            best_val = max(best_val, avg_val)
        return best_val

    def print_value_table(self, value_table, iter_num):
        print("\nIteration #{}".format(iter_num))
        for s in self.states:
            print("V_{}({}) = {}".format(iter_num, s, value_table[s]))

    def value_iteration(self, epsilon=0.01, max_iterations=100, verbose=False):
        values_old, values_new = {}, {}
        values_old = self.populate_value_table(values_old, 0.0)
        values_new = self.populate_value_table(values_new, 0.0)

        diff, num_iterations = float('inf'), 0
        while diff > epsilon and num_iterations < max_iterations:
            diff = 0.0
            for s in self.states:
                values_new[s] = self.update_value(s, values_old)
                diff = max(diff, values_new[s] - values_old[s])
            num_iterations += 1

            if verbose:
                self.print_value_table(values_new, num_iterations)

            values_old = values_new.copy()

        return values_new



