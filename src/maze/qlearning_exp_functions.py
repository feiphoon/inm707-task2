import numpy as np
from q_maze import QMaze
from e_greedy_policy import EGreedyPolicy
from qlearning import QLearning

from typing import NamedTuple, Tuple


class Results(NamedTuple):
    env_size: int
    all_rewards: float
    max_reward: int
    mean_reward: float
    var_reward: float
    all_turns_elapsed: int
    max_turns_elapsed: int
    mean_turns_elapsed: float
    var_turns_elapsed: float
    mean_ending_epsilon: float


def train_ql_for_one_episode(
    environment: QMaze, q_learning_method: QLearning, policy: EGreedyPolicy
) -> Tuple[int, int, float]:
    state: int = environment.reset()
    done: bool = False
    total_reward: float = 0

    while not done:
        next_action = policy(state, q_learning_method.q_value_store)
        next_state, reward, done = environment.step(next_action)
        next_action_index = next_action.value.index

        q_learning_method.update_q_values(
            current_state=state,
            next_action=next_action_index,
            next_reward=reward,
            next_state=next_state,
        )

        policy.update_epsilon()
        state = next_state
        total_reward += reward

    return total_reward, environment.turns_elapsed, policy.epsilon


def run_ql_experiments(
    environment: QMaze,
    policy: EGreedyPolicy,
    q_learning_method: QLearning,
    num_episodes: int,
) -> Results:
    all_rewards: list = []
    all_turns_elapsed: list = []
    all_ending_epsilon: list = []

    for _ in range(num_episodes):

        final_reward, final_turns_elapsed, final_epsilon = train_ql_for_one_episode(
            environment=environment, policy=policy, q_learning_method=q_learning_method
        )

        # Record some information
        all_rewards.append(final_reward)
        all_turns_elapsed.append(final_turns_elapsed)
        all_ending_epsilon.append(final_epsilon)

    env_size = environment.size
    max_reward = max(all_rewards)
    mean_reward = np.mean(all_rewards)
    var_reward = np.std(all_rewards)

    max_turns_elapsed = max(all_turns_elapsed)
    mean_turns_elapsed = np.mean(all_turns_elapsed)
    var_turns_elapsed = np.std(all_turns_elapsed)

    mean_ending_epsilon = np.mean(all_ending_epsilon)
    # policy.reset()

    return Results(
        env_size,
        all_rewards,
        max_reward,
        mean_reward,
        var_reward,
        all_turns_elapsed,
        max_turns_elapsed,
        mean_turns_elapsed,
        var_turns_elapsed,
        mean_ending_epsilon,
    )
