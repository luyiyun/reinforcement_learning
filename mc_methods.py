from typing import Literal

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt


def mc_policy_evaluation(
    env: gym.Env,
    policy: np.ndarray,
    num_episodes: int,
    gamma: float = 0.9,
    method: Literal["fvmc", "evmc"] = "evmc",
) -> np.ndarray:
    """
    使用首次访问蒙特卡洛方法评估给定策略的Q(s,a)函数。

    Args:
        env: OpenAI Gym 环境.
        policy: 策略数组，形状为 (env.observation_space.n, )。
        num_episodes: 用于评估的回合数。
        gamma: 折扣因子。

    Returns:
        Q: 数组，形状为 (env.observation_space.n, env.action_space.n) ，表示状态-动作值函数。
    """
    assert method in ["fvmc", "evmc"], f"Invalid method {method}"
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=float)
    N = np.zeros((env.observation_space.n, env.action_space.n), dtype=int)
    for i_episode in tqdm(range(1, num_episodes + 1), desc="Episodes: "):
        episode = []
        state, _ = env.reset()
        terminated = False
        truncated = False  # For time limits, not relevant for FrozenLake basic

        # 根据要评估的策略生成一个回合
        while not terminated and not truncated:
            # 如果策略未定义该状态的动作，随机选择 (或者可以报错)
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            if terminated or truncated:
                break
            state = next_state

        if method == "evmc":
            G = 0
            for state, action, reward in reversed(episode):
                G = reward + gamma * G
                N[state, action] += 1
                Q[state, action] += (G - Q[state, action]) / N[state, action]
        else:
            G = 0
            state_action_dict = {}
            for state, action, reward in reversed(episode):
                G = reward + gamma * G
                state_action_dict[(state, action)] = G
            for (state, action), Gi in state_action_dict.items():
                N[state, action] += 1
                Q[state, action] += (Gi - Q[state, action]) / N[state, action]
    return Q


def mc_control(
    env: gym.Env,
    num_episodes: int,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    epsilon_decay: float | None = 0.999,
    epsilon_min: float = 0.01,
    method: Literal["fvmc", "evmc"] = "evmc",
    seed: int | None = None,
):
    """
    Args:

    Returns:
        Q: 学习到的最优动作值函数。
        policy: 从Q派生出的确定性贪婪策略。
        episode_rewards: 每回合的总奖励列表，用于绘图。
    """
    assert method in ["fvmc", "evmc"], f"Invalid method {method}"
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=float)
    N = np.zeros((env.observation_space.n, env.action_space.n), dtype=int)
    reward_history = []  # 记录每回合的奖励
    for i_episode in tqdm(range(1, num_episodes + 1), desc="Episodes: "):
        episode = []
        state, _ = env.reset()
        terminated = False
        truncated = False  # For time limits, not relevant for FrozenLake basic
        episode_reward = 0
        policy = np.argmax(Q, axis=1)

        # 根据要评估的策略生成一个回合
        while not terminated and not truncated:
            # 如果策略未定义该状态的动作，随机选择 (或者可以报错)
            action = (
                env.action_space.sample()
                if rng.binomial(1, epsilon) == 1
                else policy[state]
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode.append((state, action, reward))
            if terminated or truncated:
                break
            state = next_state

        reward_history.append(episode_reward)

        G = 0
        if method == "evmc":
            for state, action, reward in reversed(episode):
                G = reward + gamma * G
                N[state, action] += 1
                Q[state, action] += (G - Q[state, action]) / N[state, action]
        else:
            state_action_dict = {}
            for state, action, reward in reversed(episode):
                G = reward + gamma * G
                state_action_dict[(state, action)] = G
            for (state, action), Gi in state_action_dict.items():
                N[state, action] += 1
                Q[state, action] += (Gi - Q[state, action]) / N[state, action]

        # Epsilon衰减 (GLIE: Greedy in the Limit of Infinite Exploration)
        if epsilon_decay is not None:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # 从Q值派生出确定性策略
    policy = np.argmax(Q, axis=1)
    return policy, Q, np.array(reward_history)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    simple_policy = np.array([2, 2, 1, 0, 2, 1, 0, 0, 2, 0, 0, 0, 2, 2, 1])
    Q = mc_policy_evaluation(env, simple_policy, num_episodes=2000, gamma=0.9)
    print(Q)
    Q = mc_policy_evaluation(
        env, simple_policy, num_episodes=2000, gamma=0.9, method="fvmc"
    )
    print(Q)

    policy, Q, episode_rewards = mc_control(
        env,
        5000,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        seed=0,
    )
    print(policy)
    print(Q)
    print(episode_rewards)
    print(np.array([["⬅️", "⬇️", "➡️", "⬆️"][i] for i in policy]).reshape(4, 4))

    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        render_mode="rgb_array_list",
    )
    observation, info = env.reset()
    episode_over = False
    while not episode_over:
        action = policy[observation]
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
    img_list = env.render()
    env.close()
    imageio.mimsave("./results/frozen_lake_mc.gif", img_list, duration=0.5, loop=0)

    # 计算reward的滑动平均
    window_size = 500
    moving_avg = np.convolve(
        episode_rewards, np.ones(window_size) / window_size, mode="valid"
    )
    plt.plot(episode_rewards, label="Episode Rewards", color="blue", alpha=0.3)
    plt.plot(
        np.arange(window_size - 1, len(episode_rewards)),
        moving_avg,
        label=f"Moving Average (window {window_size})",
        color="red",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("./results/frozen_lake_mc_history.png")
    plt.show()
