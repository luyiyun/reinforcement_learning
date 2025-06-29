# import itertools
import logging
from dataclasses import dataclass, KW_ONLY
from typing import Callable

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from base import DiscreteAgent
from utils import plot_frozen_lake_value_function, plot_rewards_history


logger = logging.getLogger(__name__)


@dataclass
class NTDAgent(DiscreteAgent):
    """
    N step TD algorithm
    """

    n: int  # step size

    # def evaluate(self, num_episodes: int, gamma: float, alpha: float) -> np.ndarray:
    #     with logging_redirect_tqdm():
    #         for e in tqdm(range(num_episodes)):
    #             state, _ = self.env.reset()
    #             done = False
    #             rewards, states = [], [state]
    #             while not done:
    #                 action = self.policy(state)
    #                 next_state, reward, terminated, truncated, _ = self.env.step(action)
    #                 done = terminated or truncated
    #                 rewards.append(reward)
    #                 states.append(next_state)

    #             logger.info(f"Episode {e}: the length of episode is {len(rewards)}")
    #             rewards = np.array(rewards)
    #             T = len(rewards)
    #             for t in range(T):
    #                 tau = t - self.n

    #                 if tau >= 0:
    #                     rewards_t = rewards[tau:t]
    #                     G = np.sum(rewards_t * (gamma ** np.arange(len(rewards_t))))

    #                     if t < T - 1:
    #                         G += (gamma**self.n) * self.V_[states[t + 1]]

    #                     self.V_[states[tau]] += alpha * (G - self.V_[states[tau]])

    #     return self.V_

    def evaluate(self, num_episodes: int, gamma: float, alpha: float) -> np.ndarray:
        for e in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            done = False
            n_rewards, n_states = [], [state]
            while not done:
                action = self.policy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                n_rewards.append(reward)
                n_states.append(next_state)

                if not done and len(n_rewards) < self.n:
                    state = next_state
                    continue

                Gn = np.sum(np.array(n_rewards) * (gamma ** np.arange(len(n_rewards))))
                if not done and len(n_rewards) >= self.n:
                    Gn += (gamma**self.n) * self.V_[next_state]
                    n_rewards.pop(0)

                update_state = n_states.pop(0)
                self.V_[update_state] += alpha * (Gn - self.V_[update_state])
                state = next_state

            while n_rewards:
                Gn = np.sum(np.array(n_rewards) * (gamma ** np.arange(len(n_rewards))))
                update_state = n_states.pop(0)
                self.V_[update_state] += alpha * (Gn - self.V_[update_state])
                n_rewards.pop(0)

        return self.V_


@dataclass
class NSARSAAgent(NTDAgent):
    def __post_init__(self):
        super().__post_init__()
        # 直接指定，不然回合Base class出现冲突
        self.policy = (
            lambda state, epsilon=0.0: self.env.action_space.sample()
            if self._rng.random() < epsilon
            else np.argmax(self.Q_[state])
        )

    def control(
        self,
        num_episodes: int,
        gamma: float,
        alpha: float,
        epsilon: float = 0.1,
        epsilon_decay: float | None = 0.999,
        epsilon_min: float = 0.01,
    ) -> np.ndarray:
        history = []
        for e in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            action = self.policy(state, epsilon)
            done = False
            n_rewards, n_states, n_actions = [], [state], [action]
            total_rewards = 0
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                n_rewards.append(reward)
                n_states.append(next_state)

                next_action = self.policy(next_state, epsilon)
                n_actions.append(next_action)

                if not done and len(n_rewards) < self.n:
                    state, action = next_state, next_action
                    total_rewards += reward
                    continue

                Gn = np.sum(np.array(n_rewards) * (gamma ** np.arange(len(n_rewards))))
                if not done and len(n_rewards) >= self.n:
                    Gn += (gamma**self.n) * self.Q_[
                        self.get_Q_index(next_state, next_action)
                    ]
                    n_rewards.pop(0)

                update_state, update_action = n_states.pop(0), n_actions.pop(0)
                self.Q_[self.get_Q_index(update_state, update_action)] += alpha * (
                    Gn - self.Q_[self.get_Q_index(update_state, update_action)]
                )
                state, action = next_state, next_action
                total_rewards += reward

            while n_rewards:
                Gn = np.sum(np.array(n_rewards) * (gamma ** np.arange(len(n_rewards))))
                update_state, update_action = n_states.pop(0), n_actions.pop(0)
                self.Q_[self.get_Q_index(update_state, update_action)] += alpha * (
                    Gn - self.Q_[self.get_Q_index(update_state, update_action)]
                )
                n_rewards.pop(0)

            # Epsilon衰减 (GLIE: Greedy in the Limit of Infinite Exploration)
            if epsilon_decay is not None:
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

            history.append(total_rewards)

        return np.array(history)


@dataclass
class NQLearningAgent(NTDAgent):
    def __post_init__(self):
        super().__post_init__()
        # 直接指定，不然回合Base class出现冲突
        self.policy = (
            lambda state, epsilon=0.0: self.env.action_space.sample()
            if self._rng.random() < epsilon
            else np.argmax(self.Qmax_[state])
        )

    def reset(self):
        super().reset()
        self.Qmax_ = np.zeros(
            shape=(*self._shape_obs, self.env.action_space.n), dtype=float
        )

    def control(
        self,
        num_episodes: int,
        gamma: float,
        alpha: float,
        epsilon: float = 0.1,
        epsilon_decay: float | None = 0.999,
        epsilon_min: float = 0.01,
    ) -> np.ndarray:
        history = []
        for e in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            action = self.policy(state, epsilon)
            done = False
            n_rewards, n_states, n_actions = [], [state], [action]
            total_rewards = 0
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                n_rewards.append(reward)
                n_states.append(next_state)

                next_action = self.policy(next_state, epsilon)
                n_actions.append(next_action)

                if not done and len(n_rewards) < self.n:
                    state, action = next_state, next_action
                    total_rewards += reward
                    continue

                Gn = np.sum(np.array(n_rewards) * (gamma ** np.arange(len(n_rewards))))
                if not done and len(n_rewards) >= self.n:
                    Gn += (gamma**self.n) * self.Qmax_[next_state].max()
                    n_rewards.pop(0)

                update_state, update_action = n_states.pop(0), n_actions.pop(0)
                self.Qmax_[self.get_Q_index(update_state, update_action)] += alpha * (
                    Gn - self.Qmax_[self.get_Q_index(update_state, update_action)]
                )
                state, action = next_state, next_action
                total_rewards += reward

            while n_rewards:
                Gn = np.sum(np.array(n_rewards) * (gamma ** np.arange(len(n_rewards))))
                update_state, update_action = n_states.pop(0), n_actions.pop(0)
                self.Qmax_[self.get_Q_index(update_state, update_action)] += alpha * (
                    Gn - self.Qmax_[self.get_Q_index(update_state, update_action)]
                )
                n_rewards.pop(0)

            # Epsilon衰减 (GLIE: Greedy in the Limit of Infinite Exploration)
            if epsilon_decay is not None:
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

            history.append(total_rewards)

        return np.array(history)


@dataclass
class LambdaTDAgent(DiscreteAgent):
    lam: float

    def evaluate(self, num_episodes: int, gamma: float, alpha: float) -> np.ndarray:
        for e in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            done = False
            E = np.zeros(self._shape_obs, dtype=float)
            while not done:
                action = self.policy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                td_error = reward + gamma * self.V_[next_state] - self.V_[state]
                E[state] += 1.0
                self.V_ += alpha * td_error * E
                E *= gamma * self.lam
                state = next_state

        return self.V_


@dataclass
class LambdaSARSAAgent(LambdaTDAgent):
    lam: float

    def __post_init__(self):
        super().__post_init__()
        # 直接指定，不然回合Base class出现冲突
        self.policy = (
            lambda state, epsilon=0.0: self.env.action_space.sample()
            if self._rng.random() < epsilon
            else np.argmax(self.Q_[state])
        )

    def control(
        self,
        num_episodes: int,
        gamma: float,
        alpha: float,
        epsilon: float = 0.1,
        epsilon_decay: float | None = 0.999,
        epsilon_min: float = 0.01,
    ) -> np.ndarray:
        history = []
        for e in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            action = self.policy(state, epsilon)
            done = False
            E = np.zeros(self._shape_obs + self._shape_act, dtype=float)
            total_rewards = 0
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_action = self.policy(next_state, epsilon)
                td_error = (
                    reward
                    + gamma * self.Q_[self.get_Q_index(next_state, next_action)]
                    - self.Q_[self.get_Q_index(state, action)]
                )
                E[self.get_Q_index(state, action)] += 1.0
                self.Q_ += alpha * td_error * E
                E *= gamma * self.lam
                state, action = next_state, next_action
                total_rewards += reward

            # Epsilon衰减 (GLIE: Greedy in the Limit of Infinite Exploration)
            if epsilon_decay is not None:
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

            history.append(total_rewards)

        return history


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    env = gym.make("FrozenLake-v1", is_slippery=True)

    # 算法参数
    num_episodes = 20000
    alpha = 0.02  # 学习率可以调小一些以获得更稳定的学习
    gamma = 0.99  # Blackjack 通常是无折扣的，因为回合总会结束

    # ===== TD(n) policy evaluation =====
    # agent = NTDAgent(env, n=2, seed=42)  # 步长为 1
    # V = agent.evaluate(num_episodes, gamma, alpha)
    # plot_frozen_lake_value_function(V, "./results/td_n_frozen_lake_value_function.png")

    # ===== n-step SARSA policy control =====
    # history_dict = {}
    # for n in range(2, 5):
    #     agent = NSARSAAgent(env, n=n, seed=42)
    #     history = agent.control(
    #         num_episodes,
    #         gamma,
    #         alpha,
    #         epsilon=1.0,
    #         epsilon_decay=0.999,
    #         epsilon_min=0.05,
    #     )
    #     history_dict[f"n={n}"] = history
    # fig, ax = plot_rewards_history(history_dict)
    # fig.savefig("./results/frozenlake_n_sarsa_history.png")

    # V = agent.evaluate(num_episodes, gamma, alpha)
    # plot_frozen_lake_value_function(
    #     V, "./results/frozenlake_n4_sarsa_value_function.png"
    # )

    # ===== n-step Q-learning policy control =====
    # history_dict = {}
    # for n in range(2, 5):
    #     agent = NQLearningAgent(env, n=n, seed=42)
    #     history = agent.control(
    #         num_episodes,
    #         gamma,
    #         alpha,
    #         epsilon=1.0,
    #         epsilon_decay=0.999,
    #         epsilon_min=0.05,
    #     )
    #     history_dict[f"n={n}"] = history
    # fig, ax = plot_rewards_history(history_dict)
    # fig.savefig("./results/frozenlake_n_QLearning_history.png")

    # V = agent.evaluate(num_episodes, gamma, alpha)
    # plot_frozen_lake_value_function(
    #     V, "./results/frozenlake_n4_QLearning_value_function.png"
    # )

    # ===== TD(lambda) policy evaluation =====
    # agent = LambdaTDAgent(env, lam=0.5, seed=42)  # 步长为 1
    # V = agent.evaluate(num_episodes, gamma, alpha)
    # plot_frozen_lake_value_function(
    #     V, "./results/td_lambda_frozen_lake_value_function.png"
    # )

    # ===== SARSA lambda policy control =====
    lambda_values = [0.1, 0.3, 0.6, 0.9]  # 不同的 lambda 值
    history_dict = {}
    for lam in lambda_values:
        agent = LambdaSARSAAgent(env, lam=lam, seed=42)
        history = agent.control(
            num_episodes,
            gamma,
            alpha,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.05,
        )
        history_dict[f"lam={lam}"] = history
    fig, ax = plot_rewards_history(history_dict)
    fig.savefig("./results/frozenlake_lambda_SARSA_history.png")

    V = agent.evaluate(num_episodes, gamma, alpha)
    plot_frozen_lake_value_function(
        V, "./results/frozenlake_lambda_sarsa_value_function.png"
    )
