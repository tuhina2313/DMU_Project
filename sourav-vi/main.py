from rl_algorithms import RLAlgorithms
from envs.oneDimGridWorld import OneDimGridWorld


def main():
    grid_world = OneDimGridWorld()
    rl_alg = RLAlgorithms(
        states=grid_world.states,
        actions=grid_world.actions,
        transitions=grid_world.transitions,
        rewards=grid_world.rewards,
        gamma=grid_world.gamma
    )

    rl_alg.value_iteration(verbose=True)


if __name__ == '__main__':
    main()