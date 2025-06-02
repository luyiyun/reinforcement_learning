from dataclasses import dataclass
from itertools import product
from typing import Callable
from typing import Literal

import numpy as np


@dataclass
class GridWorld:
    def __post_init__(self):
        self.rows = 3
        self.cols = 4
        self.n_states = self.rows * self.cols  # Total number of states
        self.n_actions = 4  # 0: Up, 1: Down, 2: Left, 3: Right

        # Grid definition (can be changed for different layouts)
        self.grid_layout = np.array(
            [
                [" ", " ", " ", "G"],  # G for Goal
                [" ", "X", " ", "T"],  # X for Obstacle, T for Trap
                ["S", " ", " ", " "],  # S for Start
            ]
        )

        # '↑', '↓', '←', '→'
        self.actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        # Rewards and penalties
        self.gamma = 0.9  # Discount factor (often considered part of the agent/solver)
        self.living_penalty = -0.04
        self.wall_penalty = -0.1
        self.obstacle_penalty = -0.1  # Penalty for hitting an obstacle
        self.goal_reward = 1.0
        self.trap_reward = -1.0

        # R(s,a,s'), 第一个维度是状态，第二个维度是动作，第三个维度是下一个状态
        self.rewards = np.zeros((self.n_states, self.n_actions, self.n_states))
        # P(s'|s,a) 第一个维度是当前状态，第二个维度是动作，第三个维度是下一个状态
        self.state_transition = np.zeros((self.n_states, self.n_actions, self.n_states))
        for r, c, a in product(
            range(self.rows), range(self.cols), range(self.n_actions)
        ):
            move = self.actions[a]
            r_next, c_next = r + move[0], c + move[1]
            state = r * self.cols + c
            if r_next < 0 or r_next >= self.rows or c_next < 0 or c_next >= self.cols:
                self.rewards[state, a, state] = self.wall_penalty
                self.state_transition[state, a, state] = 1.0
                continue
            if self.grid_layout[r_next, c_next] == "X":
                self.rewards[state, a, state] = self.obstacle_penalty
                self.state_transition[state, a, state] = 1.0
                continue
            state_next = r_next * self.cols + c_next
            self.state_transition[state, a, state_next] = 1.0
            if self.grid_layout[r_next, c_next] == "T":
                self.rewards[state, a, state_next] = self.trap_reward
            elif self.grid_layout[r_next, c_next] == "G":
                self.rewards[state, a, state_next] = self.goal_reward
            else:
                self.rewards[state, a, state_next] = self.living_penalty

    def reset(self):
        self.idx = 0
        self.current_pos = (2, 0)

    def step(self, action: int):
        if action < 0 or action >= self.n_actions:
            raise ValueError(
                f"Invalid action {action} for action space. Please choose an action from 0 to {self.n_actions - 1}."
            )

        move = self.actions[action]
        next_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
        if (
            next_pos[0] < 0
            or next_pos[0] >= self.rows
            or next_pos[1] < 0
            or next_pos[1] >= self.cols
            or self.grid_layout[next_pos[0], next_pos[1]] == "X"
        ):
            next_pos = (
                self.current_pos
            )  # Stay in the same state if it hits a wall or goes out of bounds
        reward = self.rewards[
            self.current_pos[0] * self.cols + self.current_pos[1],
            action,
            next_pos[0] * self.cols + next_pos[1],
        ]
        terminated = self.grid_layout[next_pos[0], next_pos[1]] in ["G", "T"]
        truncated = False  # No truncation condition in this simple environment

        self.current_pos = next_pos
        self.idx += 1

        return self.current_pos, reward, terminated, truncated, {}

    def play(self, reset: bool = True, agent: Callable | None = None):
        if agent is None:
            agent = lambda s: np.random.choice(self.n_actions)
        if reset:
            self.reset()
        pos_list = [self.current_pos]
        reward_list = [0]
        while True:
            pos, reward, terminated, _, _ = self.step(agent(self.current_pos))
            pos_list.append(pos)
            reward_list.append(reward)
            if terminated:
                break

        return pos_list, reward_list


def linear_algebra_policy_evaluation(
    env: GridWorld, policy: np.ndarray, flattened: bool = False
) -> np.ndarray:
    """
    policy: (n_rows, n_cols, n_actions)

    return: (n_rows, n_cols)
    """
    if not flattened:
        assert policy.shape == (env.rows, env.cols, env.n_actions), (
            "Invalid policy shape"
        )
        policy = policy.reshape(env.n_states, env.n_actions)
    else:
        assert policy.shape == (env.n_states, env.n_actions), "Invalid policy shape"
    P = np.einsum("ikj,ik->ij", env.state_transition, policy)
    R = np.einsum(
        "ikj,ik,ikj->i",
        env.rewards,
        policy,
        env.state_transition,
    )
    V = np.linalg.solve(np.eye(env.n_states) - env.gamma * P, R)
    if flattened:
        return V
    return V.reshape(env.rows, env.cols)


def dynamic_program_policy_evaluation(
    env: GridWorld, policy: np.ndarray, flattened: bool = False, theta: float = 1e-5
) -> np.ndarray:
    if not flattened:
        assert policy.shape == (env.rows, env.cols, env.n_actions), (
            "Invalid policy shape"
        )
        policy = policy.reshape(env.n_states, env.n_actions)
    else:
        assert policy.shape == (env.n_states, env.n_actions), "Invalid policy shape"
    V = np.zeros(env.n_states)
    while True:
        V_new = np.einsum(
            "ikj,ikj,ik->i", env.gamma * V + env.rewards, env.state_transition, policy
        )
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break
    if flattened:
        return V
    return V.reshape(env.rows, env.cols)


def policy_iteration(
    env: GridWorld, theta: float = 1e-5, eval_method: Literal["LA", "DP"] = "LA"
) -> np.ndarray:
    """
    因为是确定性策略，所以返回的是每个状态下对应的动作，而不是概率
    """
    policy_int = np.random.choice(env.n_actions, size=env.n_states)
    while True:
        policy = np.zeros((env.n_states, env.n_actions))
        policy[np.arange(env.n_states), policy_int] = 1
        if eval_method == "LA":
            V = linear_algebra_policy_evaluation(env, policy, flattened=True)
        elif eval_method == "DP":
            V = dynamic_program_policy_evaluation(
                env, policy, flattened=True, theta=theta
            )
        else:
            raise ValueError("Invalid evaluation method")
        Q = np.einsum("ikj,ikj->ik", env.state_transition, env.rewards + env.gamma * V)
        policy_int_new = np.argmax(Q, axis=1)
        flag_diff = (policy_int_new == policy_int).all()
        policy_int = policy_int_new
        if flag_diff:
            break
    return policy_int.reshape(env.rows, env.cols)


def value_iteration(env: GridWorld, theta: float = 1e-5) -> np.ndarray:
    V = np.zeros(env.n_states)
    while True:
        V_new = np.max(
            np.einsum(
                "ikj,ikj->ik",
                env.gamma * V + env.rewards,
                env.state_transition,
            ),
            axis=1,
        )
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            break

    Q = np.einsum("ikj,ikj->ik", env.state_transition, env.rewards + env.gamma * V)
    policy_int = np.argmax(Q, axis=1)
    return policy_int.reshape(env.rows, env.cols), V.reshape(env.rows, env.cols)


if __name__ == "__main__":
    env = GridWorld()
    pos_list, reward_list = env.play()
    print(pos_list)
    print(reward_list)

    # 随机策略
    random_policy = np.random.rand(env.rows, env.cols, env.n_actions)
    random_policy /= random_policy.sum(axis=2, keepdims=True)
    V = linear_algebra_policy_evaluation(env, random_policy)
    print("使用线性代数求解随机策略的价值函数：")
    print(V)
    V = dynamic_program_policy_evaluation(env, random_policy)
    print("使用动态规划求解随机策略的价值函数：")
    print(V)

    # 策略迭代
    policy_int_max = policy_iteration(env, eval_method="LA")
    print("策略迭代得到的策略：")
    for i in range(env.rows):
        for j in range(env.cols):
            print(["↑", "↓", "←", "→"][policy_int_max[i, j]], end=" ")
        print()
    policy_max = np.zeros((env.n_states, env.n_actions))
    policy_max[np.arange(env.n_states), policy_int_max.flatten()] = 1
    policy_max = policy_max.reshape(env.rows, env.cols, env.n_actions)
    V = linear_algebra_policy_evaluation(env, policy_max)
    print("该策略的价值函数：")
    print(V)

    # 值迭代
    policy_int_max, V_max = value_iteration(env)
    print("值迭代得到的策略：")
    for i in range(env.rows):
        for j in range(env.cols):
            print(["↑", "↓", "←", "→"][policy_int_max[i, j]], end=" ")
        print()
    print("该策略的价值函数（值迭代直接得到）：")
    print(V_max)
    policy_max = np.zeros((env.n_states, env.n_actions))
    policy_max[np.arange(env.n_states), policy_int_max.flatten()] = 1
    policy_max = policy_max.reshape(env.rows, env.cols, env.n_actions)
    V = linear_algebra_policy_evaluation(env, policy_max)
    print("该策略的价值函数（策略评估得到）：")
    print(V)
