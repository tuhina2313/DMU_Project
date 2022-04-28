import enum


class Action(enum.Enum):
    LEFT = -1
    RIGHT = 1


class OneDimGridWorld:
    """
    A simple 1x4 gridworld, mainly
    to test the VI algorithm.

    0  0  0  +10

    Assuming the last state
    to be absorbent.
    Therefore V(last) = 0

    Final V table should be

    8.1  9.0  10  0

    """
    def __init__(self):
        self.states = [i for i in range(1, 5)]

        self.actions = [a for a in Action]
        self.gamma = 0.9

        self.rewards = {}
        self._fill_rewards()

        self.transitions = {}
        self._fill_transitions()

    def _fill_rewards(self):
        for s in self.states:
            for a in self.actions:
                for s_p in self.states:
                    self.rewards[(s, a, s_p)] = 0.0

        self.rewards[(3, Action.RIGHT, 4)] = 10.0

    def _fill_transitions(self):

        # If nothing mentioned
        for s1 in self.states:
            for a in self.actions:
                for s2 in self.states:
                    self.transitions[(s1, a, s2)] = 0.0

        self.transitions[(1, Action.RIGHT, 2)] = 1.0
        self.transitions[(1, Action.LEFT, 1)] = 1.0
        self.transitions[(2, Action.RIGHT, 3)] = 1.0
        self.transitions[(2, Action.LEFT, 1)] = 1.0
        self.transitions[(3, Action.RIGHT, 4)] = 1.0
        self.transitions[(3, Action.LEFT, 2)] = 1.0
