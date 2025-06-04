import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import seaborn as sns
from matplotlib.patches import Patch


def Q_learning_control(
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
    shape_obs = [obv.n for obv in env.observation_space]
    Q = np.zeros((*shape_obs, env.action_space.n))

    rewards_history = []
    for episode in tqdm(range(n_episodes)):
        terminated, truncated = False, False
        total_episode_reward = 0

        # initial state and action
        state, _ = env.reset()
        while not (terminated or truncated):
            action = (
                env.action_space.sample()
                if rng.random() < epsilon
                else np.argmax(Q[state])
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_episode_reward += reward

            # Q learning Update Rule:
            # TD Target: R_t+1 + gamma * max_a Q(S_t+1, a)
            td_target = reward + gamma * (0 if terminated else np.max(Q[next_state]))
            # TD Error: td_target - Q(S_t, A_t)
            td_error = td_target - Q[state + (action,)]
            # Update Q(S_t, A_t)
            Q[state + (action,)] += lr * td_error

            state = next_state

        # Epsilon衰减 (GLIE: Greedy in the Limit of Infinite Exploration)
        if epsilon_decay is not None:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_history.append(total_episode_reward)

    return Q, np.array(rewards_history)


def create_grids(Q_max: np.ndarray, usable_ace: bool = False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    # state_value = defaultdict(float)
    # policy = defaultdict(int)
    # for obs, action_values in agent.q_values.items():
    #     state_value[obs] = float(np.max(action_values))
    #     policy[obs] = int(np.argmax(action_values))
    state_value = np.max(Q_max, axis=-1)
    policy = np.argmax(Q_max, axis=-1)

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


if __name__ == "__main__":
    # env = gym.make("Blackjack-v1", render_mode="human")
    # print(f"{n_states} states, {env.action_space.n} actions")
    # observation, info = env.reset()
    # episode_over = False
    # while not episode_over:
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     episode_over = terminated or truncated
    # # img_list = env.render()
    # env.close()

    # --- Train using Qlearning ---
    env = gym.make("Blackjack-v1")
    Q_max, rewards_history_q_learning = Q_learning_control(
        env, 500000, 0.01, 0.99, 1.0, 0.9, 0.05, seed=42
    )
    # Derive the optimal V-function and policy from Qlearning's Q-table
    V_optimal_ql = np.max(Q_max, axis=-1)
    optimal_policy_ql = np.argmax(Q_max, axis=-1)

    # plot the learning progress
    window_size = 500
    moving_avg = np.convolve(
        rewards_history_q_learning, np.ones(window_size) / window_size, mode="valid"
    )
    plt.plot(
        rewards_history_q_learning, label="Episode Rewards", color="blue", alpha=0.3
    )
    plt.plot(
        np.arange(window_size - 1, len(rewards_history_q_learning)),
        moving_avg,
        label=f"Moving Average (window {window_size})",
        color="red",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("./results/blackjack_qlearning_history.png")
    plt.show()

    # plot the optimal policy gif
    env = gym.make("Blackjack-v1", render_mode="rgb_array_list")
    shape_obs = [obv.n for obv in env.observation_space]
    observation, info = env.reset()
    episode_over = False
    while not episode_over:
        action = optimal_policy_ql[observation]
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
    img_list = env.render()
    env.close()
    print(f"reward: {reward}")
    imageio.mimsave(
        "./results/blackjack_qlearning_optimal.gif", img_list, duration=0.5, loop=0
    )

    # plot the optimal value function and policy
    # value_grid, policy_grid = create_grids(Q_max, usable_ace=True)
    # fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
    # fig1.savefig("./results/blackjack_qlearning_optimal_ace.png")

    # value_grid, policy_grid = create_grids(Q_max, usable_ace=False)
    # fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
    # fig2.savefig("./results/blackjack_qlearning_optimal_without_ace.png")

    # # --- Train using MC control, as a baseline ---
    # # optimal_policy_mc, Q_mc, rewards_history_mc = mc_control(
    # #     env=env,
    # #     num_episodes=10000,
    # #     gamma=0.9,
    # #     epsilon=1.0,
    # #     epsilon_decay=0.999,
    # #     epsilon_min=0.01,
    # #     seed=42,
    # # )
    # # # Derive the optimal V-function and policy from mc's Q-table
    # # V_optimal_mc = np.max(Q_mc, axis=1)
    # # print("\nOptimal V-values derived from MC's Q-values:")
    # # print(V_optimal_mc.reshape((4, 4)))
    # # print("\nLearned Policy from mc (0:Left, 1:Down, 2:Right, 3:Up):")
    # # print(np.array([["←", "↓", "→", "↑"][i] for i in optimal_policy_mc]).reshape(4, 4))

    # # # plot the learning progress
    # # window_size = 500
    # # moving_avg = np.convolve(
    # #     rewards_history_mc, np.ones(window_size) / window_size, mode="valid"
    # # )
    # # plt.plot(rewards_history_mc, label="Episode Rewards", color="blue", alpha=0.3)
    # # plt.plot(
    # #     np.arange(window_size - 1, len(rewards_history_mc)),
    # #     moving_avg,
    # #     label=f"Moving Average (window {window_size})",
    # #     color="red",
    # # )
    # # plt.xlabel("Episode")
    # # plt.ylabel("Total Reward")
    # # plt.legend()
    # # plt.savefig("./results/frozen_lake_slippery_mc_history.png")
    # # plt.show()

    # # # plot the optimal policy gif
    # # env = gym.make(
    # #     "FrozenLake-v1",
    # #     desc=None,
    # #     map_name="4x4",
    # #     is_slippery=True,
    # #     render_mode="rgb_array_list",
    # # )
    # # observation, info = env.reset()
    # # episode_over = False
    # # while not episode_over:
    # #     action = optimal_policy_mc[observation]
    # #     observation, reward, terminated, truncated, info = env.step(action)
    # #     episode_over = terminated or truncated
    # # img_list = env.render()
    # # env.close()
    # # imageio.mimsave(
    # #     "./results/frozen_lake_slippery_mc.gif", img_list, duration=0.5, loop=0
    # # )
