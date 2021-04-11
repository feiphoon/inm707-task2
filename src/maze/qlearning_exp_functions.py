import os
import copy
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
from q_maze import QMaze
from e_greedy_policy import EGreedyPolicy
from qlearning import QLearning
from utils import preprocess_hyperparameters_filename

from typing import NamedTuple, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class Results(NamedTuple):
    env_size: int
    epsilon: float
    decay: float
    gamma: float
    alpha: float
    all_rewards: float
    min_reward: int
    max_reward: int
    mean_reward: float
    var_reward: float
    all_turns_elapsed: int
    min_turns_elapsed: int
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
    hyperparameter_dict: dict,
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

    min_reward = min(all_rewards)
    max_reward = max(all_rewards)
    mean_reward = np.mean(all_rewards)
    var_reward = np.std(all_rewards)

    min_turns_elapsed = min(all_turns_elapsed)
    max_turns_elapsed = max(all_turns_elapsed)
    mean_turns_elapsed = np.mean(all_turns_elapsed)
    var_turns_elapsed = np.std(all_turns_elapsed)

    mean_ending_epsilon = np.mean(all_ending_epsilon)
    # policy.reset()

    return Results(
        env_size=hyperparameter_dict["env_size"],
        epsilon=hyperparameter_dict["epsilon"],
        decay=hyperparameter_dict["decay"],
        gamma=hyperparameter_dict["gamma"],
        alpha=hyperparameter_dict["alpha"],
        all_rewards=all_rewards,
        min_reward=min_reward,
        max_reward=max_reward,
        mean_reward=mean_reward,
        var_reward=var_reward,
        all_turns_elapsed=all_turns_elapsed,
        min_turns_elapsed=min_turns_elapsed,
        max_turns_elapsed=max_turns_elapsed,
        mean_turns_elapsed=mean_turns_elapsed,
        var_turns_elapsed=var_turns_elapsed,
        mean_ending_epsilon=mean_ending_epsilon,
    )


def train_on_qlearning(
    env_size: int,
    epsilon: float,
    decay: float,
    gamma: float,
    alpha: float,
    num_runs_tqdm: Any,
    logging_step_size: int,
    num_logged_episodes: int,
    num_episodes: int,
    output_path: str,
) -> None:

    qm = QMaze(env_size)
    eg_policy = EGreedyPolicy(epsilon=epsilon, decay=decay)
    ql = QLearning(policy=eg_policy, environment=qm, gamma=gamma, alpha=alpha)

    # Create structure to store data
    all_results: list = []

    # Plot rewards and turns elapsed
    experiment_mean_rewards: list = []
    experiment_mean_turns_elapsed: list = []

    # Collect hyperparameter information
    HYPERPARAMETER_DICT = {
        "env_size": env_size,
        "epsilon": epsilon,
        "decay": decay,
        "gamma": gamma,
        "alpha": alpha,
    }
    HYPERPARAMETER_FILE_NAME = preprocess_hyperparameters_filename(HYPERPARAMETER_DICT)
    FILE_NAME = HYPERPARAMETER_FILE_NAME
    PLOT_FILE_NAME = FILE_NAME

    # Save rendered maze
    with open(os.path.join(output_path, "maze_layout.txt"), "w") as f:
        with redirect_stdout(f):
            qm.display(debug=True)

    for run in num_runs_tqdm:
        if ((run % logging_step_size) == 0) or (run == 1):
            results = run_ql_experiments(
                environment=qm,
                policy=eg_policy,
                q_learning_method=ql,
                num_episodes=num_logged_episodes,
                hyperparameter_dict=HYPERPARAMETER_DICT,
            )
            print(
                f"Run: {run}, ending epsilon: {eg_policy.epsilon}, mean reward: {results.mean_reward}, mean turns elapsed: {results.mean_turns_elapsed}, std reward: {results.var_reward}"
            )
            all_results.append(results)
            experiment_mean_rewards.append(results.mean_reward)
            experiment_mean_turns_elapsed.append(results.mean_turns_elapsed)

    # Save detailed results
    individual_results_df = pd.DataFrame(all_results)
    individual_results_df.to_csv(
        os.path.join(output_path, HYPERPARAMETER_FILE_NAME + ".csv"), index=False
    )

    # Save mean reward plot
    sns.lineplot(data=experiment_mean_rewards)
    plt.title(f"Mean rewards over {num_episodes} episodes")
    plt.xlabel(f"Per {num_logged_episodes} episodes")
    plt.ylabel("Mean reward")
    # plt.xlim([0, LOGGING_STEP_SIZE]);

    # last_mean_reward = (NUM_RUNS, experiment_mean_rewards[-1])
    # enumerated_mean_rewards = list(enumerate(experiment_mean_rewards))
    # min_mean_reward = min(enumerated_mean_rewards, key=lambda i: i[1])
    # max_mean_reward = max(enumerated_mean_rewards, key=lambda i: i[1])

    # plt.annotate(f"Last {last_mean_reward[1]}", last_mean_reward)
    # plt.annotate(f"Min {min_mean_reward[1]}", min_mean_reward)
    # plt.annotate(f"Max {max_mean_reward[1]}", max_mean_reward)

    plt.savefig(
        os.path.join(output_path, PLOT_FILE_NAME + "_mean_rewards.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Save mean turns elapsed plot
    sns.lineplot(data=experiment_mean_turns_elapsed)
    plt.title(f"Mean turns elapsed over {num_episodes} episodes")
    plt.xlabel(f"Per {num_logged_episodes} episodes")
    plt.ylabel("Mean turns elapsed")
    # plt.xlim([0, LOGGING_STEP_SIZE]);

    # last_mean_turns_elapsed = (NUM_RUNS, experiment_mean_turns_elapsed[-1])
    # enumerated_mean_turns_elapsed = list(enumerate(experiment_mean_turns_elapsed))
    # min_mean_turns_elapsed = min(enumerated_mean_turns_elapsed, key=lambda i: i[1])
    # max_mean_turns_elapsed = max(enumerated_mean_turns_elapsed, key=lambda i: i[1])

    # plt.annotate(f"Last {last_mean_turns_elapsed[1]}", last_mean_turns_elapsed)
    # plt.annotate(f"Min {min_mean_turns_elapsed[1]}", min_mean_turns_elapsed)
    # plt.annotate(f"Max {max_mean_turns_elapsed[1]}", max_mean_turns_elapsed)

    plt.savefig(
        os.path.join(output_path, PLOT_FILE_NAME + "_mean_turns_elapsed.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Save heatmap
    results_vals = ql.display_q_values()
    plt.imshow(
        (results_vals - results_vals.min()) / (results_vals.max() - results_vals.min()),
        cmap="viridis",
    )
    plt.colorbar()
    plt.grid(False)
    plt.savefig(
        os.path.join(output_path, PLOT_FILE_NAME + "_q_values.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Write to datetime run's results summary
    results_summary = copy.deepcopy(HYPERPARAMETER_DICT)
    results_summary.update(
        {
            "experiment_mean_rewards": [experiment_mean_rewards],
            "experiment_max_rewards": max(experiment_mean_rewards),
            "experiment_mean_turns_elapsed": [experiment_mean_turns_elapsed],
            "experiment_min_turns_elapsed": min(experiment_mean_turns_elapsed),
        }
    )
    results_summary_df = pd.DataFrame.from_dict(results_summary)
    # Append result summary to overall file.
    # Creates this csv if doesn't exist,
    # adds header if being created, otherwise skip.
    with open(os.path.join(output_path, "results_summary.csv"), "a") as f:
        results_summary_df.to_csv(f, header=f.tell() == 0, index=False)
