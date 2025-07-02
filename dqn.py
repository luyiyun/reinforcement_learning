from dataclasses import dataclass
from collections import deque
import random
import os.path as osp
import os

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import pandas as pd


class DQNNet(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hiddens: tuple[int, ...],
        bn: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        if bn:
            layers.append(nn.BatchNorm1d(n_inputs))
        for i, o in zip([n_inputs] + list(hiddens)[:-1], hiddens):
            layers.append(nn.Linear(i, o))
            if bn:
                layers.append(nn.BatchNorm1d(o))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hiddens[-1], n_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class ReplayBuffer:
    """
    一个固定大小的缓冲区，用于存储经验元组。
    """

    capacity: int

    def __post_init__(self):
        # 使用双端队列作为底层存储，设定最大容量
        self.buffer = deque(maxlen=self.capacity)
        self.state_is_array = None

    def push(
        self,
        state: int | np.ndarray,
        action: int,
        reward: float,
        next_state: int | np.ndarray,
        done: bool,
    ):
        """将一个经验元组存入缓冲区。"""
        # 为了节省内存，状态通常存储为 uint8 类型
        if self.state_is_array is None:
            self.state_is_array = isinstance(state, np.ndarray)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """从缓冲区中随机采样一个批次的经验。"""
        # 从缓冲区中随机选择 batch_size 个样本
        samples = random.sample(self.buffer, batch_size)

        # 将样本解压并堆叠成批次
        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.stack(states, axis=0) if self.state_is_array else np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0)
            if self.state_is_array
            else np.array(next_states),
            np.array(dones, dtype=bool),
        )

    def __len__(self):
        """返回当前缓冲区中的样本数量。"""
        return len(self.buffer)


@dataclass
class DQN:
    hiddens: tuple[int, ...] = (64, 64)
    bn: bool = False
    dropout: float = 0.0
    learning_rate: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 10000
    target_update_frequency: int = 5
    n_episodes: int = 2000
    max_steps_per_episode: int = 1000
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 128
    device: str = "cpu"
    soft_update_tau: float | None = None
    early_stop_target: float | None = None
    early_stop_interval: int = 100
    double_dqn: bool = False

    def _get_n_inputs(self, space: gym.Space) -> int:
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return np.prod(space.shape)
        elif isinstance(space, gym.spaces.Tuple):
            return sum(self._get_n_inputs(s) for s in space.spaces)
        else:
            raise ValueError(f"Unsupported space: {space}")

    def __call__(
        self, env: gym.Env, state: int | np.ndarray, epsilon: float = 0.0
    ) -> int:
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            if isinstance(state, np.ndarray):
                state = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
            else:
                raise NotImplementedError
            self.online_net.eval()
            with torch.no_grad():
                q_values = self.online_net(state)
            return q_values.argmax().item()

    def update_target_net(self):
        if self.soft_update_tau is None:
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            for target_param, online_param in zip(
                self.target_net.parameters(), self.online_net.parameters()
            ):
                target_param.data.copy_(
                    (1 - self.soft_update_tau) * target_param.data
                    + self.soft_update_tau * online_param.data
                )

    def train(
        self, batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> float:
        # 从缓冲区采样
        states, actions, rewards, next_states, dones = batch

        # 将 numpy 数组转换为 torch 张量
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.bool)

        # 1. 计算当前Q值 (Predicted Q-values)
        #    网络输出所有动作的Q值，我们只关心已执行动作的Q值
        q_pred_all = self.online_net(states.reshape(-1, self._n_inputs))
        # 使用 gather 从 (N, num_actions) 中选取 (N, 1) 的特定动作的Q值
        q_pred = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. 计算目标Q值 (Target Q-values)
        inpt = next_states.reshape(-1, self._n_inputs)
        with torch.no_grad():  # 目标值的计算不涉及梯度
            if self.double_dqn:
                # 使用 online_net 选择下一个状态的动作，再使用 target_net 计算下一个状态的Q值
                next_actions = self.online_net(inpt).argmax(dim=1)
                next_q_value = (
                    self.target_net(inpt)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )
            else:
                # 使用目标网络计算下一个状态的所有动作的Q值
                next_q_values = self.target_net(inpt)
                # 选择其中最大的Q值作为下一个状态的价值
                next_q_value = next_q_values.max(dim=1)[0]
            # 如果是终止状态，则下一个状态的价值为0
            next_q_value[dones] = 0.0
            # 计算贝尔曼目标
            y_target = rewards + self.gamma * next_q_value

        # 3. 计算损失
        loss = self.loss_fn(q_pred, y_target)

        # 4. 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        # 5. 定期更新目标网络
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.update_target_net()

        return loss.item()

    def fit(self, env: gym.Env):
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            "DQN only supports Discrete action spaces"
        )
        self._n_inputs = self._get_n_inputs(env.observation_space)
        self._n_outputs = env.action_space.n

        self.online_net = DQNNet(
            self._n_inputs, self._n_outputs, self.hiddens, self.bn, self.dropout
        ).to(self.device)
        self.target_net = DQNNet(
            self._n_inputs, self._n_outputs, self.hiddens, self.bn, self.dropout
        ).to(self.device)
        self.update_target_net()
        self.target_net.eval()
        self.online_net.train()

        # 定义优化器和损失函数
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.learn_step_counter = 0
        self.history = {"loss": [], "total_reward": []}

        # 主训练循环
        epsilon = self.epsilon_start
        for episode in tqdm(range(1, self.n_episodes + 1)):
            state, _ = env.reset()
            episode_reward, episode_loss = 0, 0
            for step in range(1, self.max_steps_per_episode + 1):
                action = self(env, state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.replay_buffer.push(state, action, reward, next_state, done)

                # 当缓冲区有足够数据时开始学习
                if len(self.replay_buffer) > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    loss = self.train(batch)
                    episode_loss += loss

                episode_reward += reward
                state = next_state
                if done:
                    break

            episode_loss /= step
            tqdm.write(
                f"Episode {episode}: reward={episode_reward}, loss={episode_loss}, epsilon={epsilon}"
            )
            self.history["total_reward"].append(episode_reward)
            self.history["loss"].append(episode_loss)

            if self.early_stop_target is not None:
                score = np.mean(
                    self.history["total_reward"][-self.early_stop_interval :]
                )
                if episode > 100 and score >= self.early_stop_target:
                    tqdm.write(f"Early stop at episode {episode}, score={score}")
                    break

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

        env.close()


if __name__ == "__main__":
    results_root = "./results/doubleDQN"
    os.makedirs(results_root, exist_ok=True)

    env = gym.make("LunarLander-v3")
    model = DQN(
        device="cuda",
        soft_update_tau=1e-3,
        n_episodes=2000,
        early_stop_target=250,
        double_dqn=True,
    )
    model.fit(env)
    pd.DataFrame(model.history).to_csv(osp.join(results_root, "history.csv"))

    # plot the total reward and loss history
    historys = model.history
    window_size = 50
    fig, axs = plt.subplots(ncols=len(historys), figsize=(4 * len(historys), 4))
    for i, (key, arri) in enumerate(historys.items()):
        axs[i].plot(arri, label="True values", alpha=0.3)
        moving_avg = np.convolve(arri, np.ones(window_size) / window_size, mode="valid")
        axs[i].plot(
            np.arange(window_size - 1, len(arri)),
            moving_avg,
            label="Moving average",
        )
        axs[i].set_xlabel("Episode")
        axs[i].set_ylabel(key)
        axs[i].legend()
    fig.tight_layout()
    fig.savefig(osp.join(results_root, "lunar_lander_dqn_history.png"))

    # generate a video of the agent playing the game
    env = gym.make("LunarLander-v3", render_mode="rgb_array_list")
    img_list = []
    for i in range(5):
        state, _ = env.reset()
        while True:
            action = model(env, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            state = next_state
        img_list.extend(env.render())
    env.close()
    imageio.mimsave(
        osp.join(results_root, "lunar_lander_dqn_video.gif"),
        img_list,
        duration=0.5,
        loop=0,
    )
