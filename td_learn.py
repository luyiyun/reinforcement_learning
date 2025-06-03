import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from mc_methods import mc_control


def td0_policy_evaluation(
    env: gym.Env, policy: np.ndarray, n_episodes: int, lr: float, gamma: float
):
    V = np.zeros(env.observation_space.n)
    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = policy[state]  # Get action from the fixed policy
            next_state, reward, terminated, truncated, _ = env.step(action)

            # TD Target: R_t+1 + gamma * V(S_t+1)
            # If next_state is terminal, its value is 0.
            td_target = reward + gamma * V[next_state] * (1 - terminated)
            # TD Error: td_target - V(S_t)
            td_error = td_target - V[state]
            # Update V(S_t)
            V[state] += lr * td_error

            state = next_state
    return V


def sarsa_control(
    env: gym.Env,
    n_episodes: int,
    lr: float,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    epsilon_decay: float | None = 0.999,
    epsilon_min: float = 0.01,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Q = np.zeros(
        (env.observation_space.n, env.action_space.n)
    )  # Initialize Q-table with zeros

    rewards_history = []
    for episode in tqdm(range(n_episodes)):
        terminated, truncated = False, False
        total_episode_reward = 0

        # initial state and action
        state, _ = env.reset()
        action = (
            env.action_space.sample() if rng.random() < epsilon else np.argmax(Q[state])
        )
        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_episode_reward += reward

            next_action = (
                env.action_space.sample()
                if rng.random() < epsilon
                else np.argmax(Q[next_state])
            )

            # SARSA Update Rule:
            # TD Target: R_t+1 + gamma * Q(S_t+1, A_t+1)
            # If next_state is terminal, Q(next_state, next_action) is 0
            td_target = reward + gamma * Q[next_state, next_action] * (1 - terminated)
            # TD Error: td_target - Q(S_t, A_t)
            td_error = td_target - Q[state, action]
            # Update Q(S_t, A_t)
            Q[state, action] += lr * td_error

            state = next_state
            action = next_action

        # Epsilon衰减 (GLIE: Greedy in the Limit of Infinite Exploration)
        if epsilon_decay is not None:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_history.append(total_episode_reward)

    return Q, np.array(rewards_history)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True)

    # --- Evaluate the fixed policy using TD(0) ---
    fixed_policy = np.full((4, 4), 2)
    fixed_policy[:, -1] = 1
    fixed_policy = fixed_policy.flatten()
    V_eval = td0_policy_evaluation(env, fixed_policy, 5000, 0.1, 0.9)
    print(V_eval.reshape(4, 4))

    # --- Train using SARSA ---
    Q_sarsa, rewards_history_sarsa = sarsa_control(
        env, 10000, 0.1, 0.9, 1.0, 0.999, 0.01, seed=42
    )
    # Derive the optimal V-function and policy from SARSA's Q-table
    V_optimal_sarsa = np.max(Q_sarsa, axis=1)
    optimal_policy_sarsa = np.argmax(Q_sarsa, axis=1)
    print("\nOptimal V-values derived from SARSA's Q-values:")
    print(V_optimal_sarsa.reshape((4, 4)))
    print("\nLearned Policy from SARSA (0:Left, 1:Down, 2:Right, 3:Up):")
    print(
        np.array([["←", "↓", "→", "↑"][i] for i in optimal_policy_sarsa]).reshape(4, 4)
    )

    # plot the learning progress
    window_size = 500
    moving_avg = np.convolve(
        rewards_history_sarsa, np.ones(window_size) / window_size, mode="valid"
    )
    plt.plot(rewards_history_sarsa, label="Episode Rewards", color="blue", alpha=0.3)
    plt.plot(
        np.arange(window_size - 1, len(rewards_history_sarsa)),
        moving_avg,
        label=f"Moving Average (window {window_size})",
        color="red",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("./results/frozen_lake_sarsa_history.png")
    plt.show()

    # plot the optimal policy gif
    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=True,
        render_mode="rgb_array_list",
    )
    observation, info = env.reset()
    episode_over = False
    while not episode_over:
        action = optimal_policy_sarsa[observation]
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
    img_list = env.render()
    env.close()
    imageio.mimsave("./results/frozen_lake_sarsa.gif", img_list, duration=0.5, loop=0)

    # --- Train using MC control, as a baseline ---
    # optimal_policy_mc, Q_mc, rewards_history_mc = mc_control(
    #     env=env,
    #     num_episodes=10000,
    #     gamma=0.9,
    #     epsilon=1.0,
    #     epsilon_decay=0.999,
    #     epsilon_min=0.01,
    #     seed=42,
    # )
    # # Derive the optimal V-function and policy from mc's Q-table
    # V_optimal_mc = np.max(Q_mc, axis=1)
    # print("\nOptimal V-values derived from MC's Q-values:")
    # print(V_optimal_mc.reshape((4, 4)))
    # print("\nLearned Policy from mc (0:Left, 1:Down, 2:Right, 3:Up):")
    # print(np.array([["←", "↓", "→", "↑"][i] for i in optimal_policy_mc]).reshape(4, 4))

    # # plot the learning progress
    # window_size = 500
    # moving_avg = np.convolve(
    #     rewards_history_mc, np.ones(window_size) / window_size, mode="valid"
    # )
    # plt.plot(rewards_history_mc, label="Episode Rewards", color="blue", alpha=0.3)
    # plt.plot(
    #     np.arange(window_size - 1, len(rewards_history_mc)),
    #     moving_avg,
    #     label=f"Moving Average (window {window_size})",
    #     color="red",
    # )
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.legend()
    # plt.savefig("./results/frozen_lake_slippery_mc_history.png")
    # plt.show()

    # # plot the optimal policy gif
    # env = gym.make(
    #     "FrozenLake-v1",
    #     desc=None,
    #     map_name="4x4",
    #     is_slippery=True,
    #     render_mode="rgb_array_list",
    # )
    # observation, info = env.reset()
    # episode_over = False
    # while not episode_over:
    #     action = optimal_policy_mc[observation]
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     episode_over = terminated or truncated
    # img_list = env.render()
    # env.close()
    # imageio.mimsave(
    #     "./results/frozen_lake_slippery_mc.gif", img_list, duration=0.5, loop=0
    # )
