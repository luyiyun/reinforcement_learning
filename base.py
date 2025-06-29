from dataclasses import dataclass, KW_ONLY
from typing import Callable

import numpy as np
import gymnasium as gym


@dataclass
class DiscreteAgent:
    env: gym.Env
    _: KW_ONLY
    policy: Callable | None = None
    seed: int | None = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._shape_obs = (
            [obs.n for obs in self.env.observation_space]
            if isinstance(self.env.observation_space, gym.spaces.Tuple)
            else [self.env.observation_space.n]
        )
        self._shape_act = (
            [act.n for act in self.env.action_space]
            if isinstance(self.env.action_space, gym.spaces.Tuple)
            else [self.env.action_space.n]
        )
        if self.policy is None:
            self.policy = lambda obs: self.env.action_space.sample()
        self.reset()

    def get_Q_index(
        self, state: tuple[int, ...] | int, action: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        if not isinstance(state, tuple):
            state = (state,)
        if not isinstance(action, tuple):
            action = (action,)
        return (*state, *action)

    def evaluate(self):
        """policy evaluation"""
        raise NotImplementedError

    def control(self):
        """policy control"""
        raise NotImplementedError

    def reset(self):
        # self.Q_ = self._rng.standard_normal(self._shape_obs + self._shape_act)
        # self.V_ = self._rng.standard_normal(self._shape_obs)
        self.Q_ = np.zeros(self._shape_obs + self._shape_act, dtype=float)
        self.V_ = np.zeros(self._shape_obs, dtype=float)


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    print(env.observation_space)
    print(env.action_space)
    print(type(env.observation_space))

    env = gym.make("FrozenLake-v1")
    print(env.observation_space)
    print(env.action_space)

    env = gym.make("Blackjack-v1")
    print(env.observation_space)
    print(env.action_space)
