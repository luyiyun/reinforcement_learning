from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Bandit:
    K: int
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def step(self, k: int) -> float:
        """Returns the reward for taking action k."""
        raise NotImplementedError


@dataclass
class BernoilliBandit(Bandit):
    def __post_init__(self):
        super().__post_init__()
        self.prob = self.rng.uniform(size=self.K)
        self.best_action = np.argmax(self.prob)
        self.best_prob = self.prob[self.best_action]

    def step(self, k: int) -> float:
        return 1 if self.rng.random() < self.prob[k] else 0

    def print(self):
        print(f"Probability of each arm: {self.prob}")


@dataclass
class Solver:
    bandit: BernoilliBandit
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.counts = np.zeros(self.bandit.K, dtype=int)
        self.regret = 0.0
        self.action = []
        self.regrets = []

    def run_one_step(self) -> int:
        """select an action, decide by the specific solver."""
        raise NotImplementedError

    def update_regret(self, k: int):
        self.regret += self.bandit.best_prob - self.bandit.prob[k]
        self.regrets.append(self.regret)

    def run(self, T: int):
        for _ in range(T):
            k = self.run_one_step()
            self.counts[k] += 1
            self.update_regret(k)
            self.action.append(k)


@dataclass
class SolverWithEstimators(Solver):
    init_prob: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.estimators = np.full(self.bandit.K, self.init_prob)

    def _update_estimators(self, k: int):
        # update estimators
        r = self.bandit.step(k)
        self.estimators[k] += (r - self.estimators[k]) / (self.counts[k] + 1)


@dataclass
class EpsilonGreedy(SolverWithEstimators):
    epsilon: float = 0.01
    epsilon_decay: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.total_counts = 0

    def run_one_step(self) -> int:
        self.total_counts += 1
        current_epsilon = (
            self.epsilon / self.total_counts if self.epsilon_decay else self.epsilon
        )
        if self.rng.random() < current_epsilon:
            k = self.rng.choice(self.bandit.K)
        else:
            k = np.argmax(self.estimators)

        self._update_estimators(k)
        return k


@dataclass
class UCB(SolverWithEstimators):
    p: float = 0.01
    c: float = 1.0
    p_decay: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.total_counts = 0

    def run_one_step(self) -> int:
        self.total_counts += 1
        current_p = self.p / self.total_counts if self.p_decay else self.p
        ucb = self.estimators + self.c * np.sqrt(
            -0.5 * np.log(current_p) / (self.counts + 1)
        )
        k = np.argmax(ucb)
        self._update_estimators(k)
        return k


@dataclass
class ThompsonSampling(Solver):
    a_prior: float = 1.0
    b_prior: float = 1.0
    method: Literal["sample", "mean", "mode"] = "sample"

    def __post_init__(self):
        super().__post_init__()
        self.a = np.full(self.bandit.K, self.a_prior)
        self.b = np.full(self.bandit.K, self.b_prior)

    def run_one_step(self) -> int:
        if self.method == "sample":
            estimator = self.rng.beta(self.a, self.b)
        elif self.method == "mean":
            estimator = self.a / (self.a + self.b)
        elif self.method == "mode":
            estimator = (self.a - 1) / (self.a + self.b - 2)
        else:
            raise ValueError("Invalid method")

        k = np.argmax(estimator)
        r = self.bandit.step(k)
        self.a[k] += r
        self.b[k] += 1 - r
        return k


def plot_results(
    solvers: dict[str, Solver], save_fn: str | None = None, log_scale: bool = False
):
    fig, ax = plt.subplots()
    for name, solver in solvers.items():
        ax.plot(solver.regrets, label=name)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Regrets")
    ax.set_title(f"{solver.bandit.K}-armed bandit")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    if save_fn is not None:
        fig.savefig(save_fn)
    plt.show()


def main():
    bandit = BernoilliBandit(K=10, seed=42)
    bandit.print()

    T = 10000
    solvers = {}

    # epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    # epsilons = [1e-4, 0.01]
    # epsilons = [0.01]
    # for epsilon in epsilons:
    #     epsilon_greedy_solver = EpsilonGreedy(bandit, epsilon=epsilon, seed=42)
    #     epsilon_greedy_solver.run(T=T)
    #     solvers[f"epsilon-greedy($\epsilon$={epsilon})"] = epsilon_greedy_solver
    #
    # decay_eps_greedy_solver = EpsilonGreedy(
    #     bandit, epsilon=0.01, epsilon_decay=True, seed=42
    # )
    # decay_eps_greedy_solver.run(T=T)
    # solvers["decaying epsilon-greedy(initial $\epsilon$=0.01)"] = (
    #     decay_eps_greedy_solver
    # )

    # ucb_solver = UCB(bandit, p=0.01, c=1.0, seed=42)
    # ucb_solver.run(T=T)
    # solvers["UCB(initial $p$=0.01, $c$=1.0)"] = ucb_solver
    # ucb_solver = UCB(bandit, p=0.1, c=1.0, seed=42)
    # ucb_solver.run(T=T)
    # solvers["UCB(initial $p$=0.1, $c$=1.0)"] = ucb_solver
    # ucb_solver = UCB(bandit, p=0.1, c=1.0, p_decay=True, seed=42)
    # ucb_solver.run(T=T)
    # solvers["p-decaying UCB(initial $p$=0.1, $c$=1.0)"] = ucb_solver

    for m in ["sample", "mean", "mode"]:
        thompson_sampling_solver = ThompsonSampling(bandit, seed=42, method=m)
        thompson_sampling_solver.run(T=T)
        solvers[f"Thompson Sampling({m})"] = thompson_sampling_solver
    # thompson_sampling_solver = ThompsonSampling(bandit, seed=42)
    # thompson_sampling_solver.run(T=T)
    # solvers["Thompson Sampling"] = thompson_sampling_solver

    plot_results(solvers, "./results/Figure_thompson2.png", log_scale=True)


if __name__ == "__main__":
    main()
