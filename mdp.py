from dataclasses import dataclass
from typing import Self, Literal

import numpy as np


@dataclass
class MRP:
    states: np.ndarray
    probabilities: np.ndarray
    rewards: np.ndarray
    gamma: float

    def __post_init__(self):
        assert (
            self.states.shape[0]
            == self.probabilities.shape[0]
            == self.probabilities.shape[1]
            == self.rewards.shape[0]
        )
        assert self.gamma >= 0 and self.gamma <= 1

    def values(self) -> np.ndarray:
        return (
            np.linalg.inv(np.eye(len(self.states)) - self.gamma * self.probabilities)
            @ self.rewards
        )


@dataclass
class MDP:
    states: np.ndarray  # nS
    probabilities: np.ndarray  # nS x nA x nS
    rewards: np.ndarray  # nS x nA
    gamma: float
    actions: np.ndarray  # nA
    seed: int | None = None

    def __post_init__(self):
        assert (
            self.states.shape[0]
            == self.probabilities.shape[0]
            == self.probabilities.shape[2]
            == self.rewards.shape[0]
        )
        assert (
            self.actions.shape[0]
            == self.rewards.shape[1]
            == self.probabilities.shape[1]
        )
        assert self.gamma >= 0 and self.gamma <= 1

        self.rng = np.random.default_rng(self.seed)

    def set_policy(self, policy: np.ndarray) -> Self:
        self.policy = policy  # nS x nA
        assert self.states.shape[0] == self.policy.shape[0]
        assert self.actions.shape[0] == self.policy.shape[1]

    def to_mrp(self) -> MRP:
        return MRP(
            states=self.states,
            probabilities=np.einsum("ikj,ik->ij", self.probabilities, self.policy),
            rewards=(self.rewards * self.policy).sum(axis=1),
            gamma=self.gamma,
        )

    def values(
        self,
        method: Literal["inv", "mc"] = "inv",
        mc_nsamples: int = 1000,
        mc_max_steps: int = 20,
    ) -> np.ndarray:
        if method == "inv":
            return self.to_mrp().values()

        episode_indices_sa, episode_indices_r = self.sample_idx(
            mc_nsamples, max_steps=mc_max_steps
        )

        V, N = np.zeros(len(self.states)), np.zeros(len(self.states))
        for i in range(mc_nsamples):
            episode_sa_idx = episode_indices_sa[i]
            episode_r_idx = episode_indices_r[i]
            mask = ~np.isnan(episode_r_idx)
            episode_s_idx = episode_sa_idx[:-1][mask, 0]
            episode_r_idx = episode_r_idx[mask]

            G = 0
            for s, r in zip(episode_s_idx[::-1], episode_r_idx[::-1]):
                G = r + self.gamma * G
                N[s] += 1
                V[s] += (G - V[s]) / N[s]

        return V

    def sample_idx(
        self,
        n_samples: int,
        state_idx: int | None = None,
        max_steps: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        res_sa = np.full((n_samples, max_steps + 1, 2), -1, dtype=int)
        res_r = np.full((n_samples, max_steps), np.nan, dtype=float)
        for i in range(n_samples):
            s = state_idx or self.rng.choice(len(self.states))
            res_sa[i, 0, 0] = s
            for step in range(max_steps):
                if s == len(self.states) - 1:
                    # terminal state is the last state
                    break
                a = self.rng.choice(len(self.actions), p=self.policy[s])
                r = self.rewards[s, a]
                s = self.rng.choice(len(self.states), p=self.probabilities[s, a])
                res_sa[i, step, 1] = a
                res_sa[i, step + 1, 0] = s
                res_r[i, step] = r

        return res_sa, res_r

    def print_episode(self, episode_sa_idx: np.ndarray, episode_r_idx: np.ndarray):
        res = []
        for i in range(episode_sa_idx.shape[0]):
            s, a = episode_sa_idx[i]
            r = episode_r_idx[i]
            if s == -1:
                break
            res_i = [f"S:{self.states[s]}"]
            if a != -1:
                res_i.append(f"A:{self.actions[a]}")
            if not np.isnan(r):
                res_i.append(f"R:{r}")
            res.append(",".join(res_i))
        print("->".join(res))


if __name__ == "__main__":
    # mrp = MRP(
    #     states=np.arange(6),
    #     probabilities=np.array(
    #         [
    #             [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    #             [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    #             [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    #             [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    #             [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    #             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    #         ]
    #     ),
    #     rewards=np.array([-1, -2, -2, 10, 1, 0]),
    #     gamma=0.5,
    # )
    # print(mrp.values())

    P = np.zeros((5, 7, 5))
    P[0, 0, 0] = 1.0
    P[0, 2, 1] = 1.0
    P[1, 1, 0] = 1.0
    P[1, 3, 2] = 1.0
    P[2, 4, 3] = 1.0
    P[2, 5, 4] = 1.0
    P[3, 5, 4] = 1.0
    P[3, 6, 1] = 0.2
    P[3, 6, 2] = 0.4
    P[3, 6, 3] = 0.4
    R = np.zeros((5, 7))
    R[0, 0] = -1
    R[0, 2] = 0
    R[1, 1] = -1
    R[1, 3] = -2
    R[2, 4] = -2
    R[2, 5] = 0
    R[3, 5] = 10
    R[3, 6] = 1
    mdp = MDP(
        states=np.arange(5),
        actions=np.array(
            ["保持1", "前往1", "前往2", "前往3", "前往4", "前往5", "概率前往"]
        ),
        probabilities=P,
        rewards=R,
        gamma=0.5,
    )

    policy1 = np.zeros((5, 7))
    policy1[[0, 0, 1, 1, 2, 2, 3, 3], [0, 2, 1, 3, 4, 5, 5, 6]] = 0.5
    mdp.set_policy(policy1)
    print("the values of policy 1: ", mdp.values())

    # policy2 = np.zeros((5, 7))
    # policy2[0, 0] = 0.6
    # policy2[0, 2] = 0.4
    # policy2[1, 1] = 0.3
    # policy2[1, 3] = 0.7
    # policy2[2, 4] = 0.5
    # policy2[2, 5] = 0.5
    # policy2[3, 5] = 0.1
    # policy2[3, 6] = 0.9
    # mdp.set_policy(policy2)
    # print("the values of policy 2: ", mdp.values())

    mdp.set_policy(policy1)
    episode_indices_sa, episode_indices_r = mdp.sample_idx(5, None, 20)
    # print(episode_indices)
    mdp.print_episode(episode_indices_sa[0], episode_indices_r[0])
    mdp.print_episode(episode_indices_sa[1], episode_indices_r[1])
    mdp.print_episode(episode_indices_sa[2], episode_indices_r[2])
    print("the values of policy 1 (using mc): ", mdp.values("mc"))
