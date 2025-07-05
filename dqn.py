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


def get_mlp(
    n_inputs: int,
    n_outputs: int,
    hiddens: tuple[int, ...],
    bn: bool = False,
    dp: float = 0.0,
    last_only_linear: bool = True,
) -> nn.Module:
    layers = []
    if bn:
        layers.append(nn.BatchNorm1d(n_inputs))
    for i, o in zip([n_inputs] + list(hiddens)[:-1], hiddens):
        layers.append(nn.Linear(i, o))
        if bn:
            layers.append(nn.BatchNorm1d(o))
        layers.append(nn.ReLU())
        if dp > 0.0:
            layers.append(nn.Dropout(dp))
    layers.append(nn.Linear(hiddens[-1], n_outputs))
    if not last_only_linear:
        if bn:
            layers.append(nn.BatchNorm1d(n_outputs))
        layers.append(nn.ReLU())
        if dp > 0.0:
            layers.append(nn.Dropout(dp))
    return nn.Sequential(*layers)


class DQNNet(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hiddens: tuple[int, ...],
        dueling_hiddens: tuple[int, ...] = (),
        bn: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._flag_dueling = bool(dueling_hiddens)
        self.encoder = get_mlp(
            n_inputs, hiddens[-1], hiddens[:-1], bn, dropout, last_only_linear=False
        )
        if self._flag_dueling:
            self.value_head = get_mlp(
                hiddens[-1], 1, dueling_hiddens, bn, dropout, last_only_linear=True
            )
            self.advantage_head = get_mlp(
                hiddens[-1],
                n_outputs,
                dueling_hiddens,
                bn,
                dropout,
                last_only_linear=True,
            )
        else:
            self.advantage_head = nn.Linear(hiddens[-1], n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self._flag_dueling:
            v = self.value_head(z)
            a = self.advantage_head(z)
            q = v + (a - a.mean(dim=1, keepdim=True))
        else:
            q = self.advantage_head(z)
        return q


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

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """从缓冲区中随机采样一个批次的经验。"""
        # 从缓冲区中随机选择 batch_size 个样本
        samples = random.sample(self.buffer, batch_size)

        # 将样本解压并堆叠成批次
        states, actions, rewards, next_states, dones = zip(*samples)

        return {
            "states": np.stack(states, axis=0)
            if self.state_is_array
            else np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.stack(next_states, axis=0)
            if self.state_is_array
            else np.array(next_states),
            "dones": np.array(dones, dtype=bool),
        }

    def __len__(self):
        """返回当前缓冲区中的样本数量。"""
        return len(self.buffer)


@dataclass
class SumTree:
    """
    SumTree 数据结构，用于实现高效的带优先级采样。
    它不是一个标准的面向对象的树实现，而是使用一个数组来表示树结构，以提高效率。
    Tree structure and array storage:
    Tree:
            0
            |
        +----+----+
        |         |
        1         2
        |         |
        +--+--+   +--+--+
        |     |   |     |
        3     4   5     6
    Array: [0, 1, 2, 3, 4, 5, 6]
    索引关系:
    - 父节点索引: (i - 1) // 2
    - 左子节点索引: 2 * i + 1
    - 右子节点索引: 2 * i + 2
    """

    capacity: int

    def __post_init__(self):
        # 树的节点总数 = 2 * capacity - 1
        # 例如 capacity=4, 节点数为7 (0-6)
        # 节点 0,1,2 为非叶子节点，节点 3,4,5,6 为叶子节点
        self.tree = np.zeros(2 * self.capacity - 1)
        # 实际数据存储在叶子节点，从 tree[capacity - 1] 开始
        self.write_ptr = 0  # 指向下一个要写入的叶子节点的索引

    def _propagate(self, idx: int, change: float):
        """从指定索引向上更新父节点的值。"""
        parent_idx = (idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate(parent_idx, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """根据采样值 s 查找对应的叶子节点索引。"""
        left_child_idx = 2 * idx + 1
        right_child_idx = left_child_idx + 1

        # 如果到达叶子节点，则返回当前索引
        if left_child_idx >= len(self.tree):
            return idx

        if s <= self.tree[left_child_idx]:
            return self._retrieve(left_child_idx, s)
        else:
            return self._retrieve(right_child_idx, s - self.tree[left_child_idx])

    @property
    def total_p(self) -> float:
        """返回所有优先级的总和（即根节点的值）。"""
        return self.tree[0]

    def add(self, p: float):
        """
        向树中添加一个新的优先级。
        总是从 self.write_ptr 指向的位置开始写入，实现循环缓冲区的效果。
        """
        # 找到要写入的叶子节点的索引
        tree_idx = self.write_ptr + self.capacity - 1

        # 更新树
        self.update(tree_idx, p)

        # 更新写入指针
        self.write_ptr = (self.write_ptr + 1) % self.capacity

    def update(self, tree_idx: int, p: float):
        """根据树的索引更新一个叶子节点的优先级。"""
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def get_leaf(self, s: float) -> tuple[int, float]:
        """
        根据采样值 s 获取叶子节点。
        返回: (叶子节点的索引, 该叶子的优先级值)
        """
        idx = self._retrieve(0, s)
        priority = self.tree[idx]
        return idx, priority


@dataclass
class PrioritizedReplayBuffer:
    capacity: int
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment_per_sampling: float = 0.001
    epsilon = 1e-5  # 防止TD误差为0时优先级为0
    max_priority = 1.0  # 新经验的初始优先级，确保它们至少有一次被采样的机会

    def __post_init__(self):
        """
        初始化优先经验回放缓冲区。

        Args:
            capacity (int): 缓冲区最大容量。
            alpha (float): 优先级指数。alpha=0表示均匀采样，alpha=1表示完全按优先级采样。
            beta (float): 重要性采样权重指数。beta=0表示无校正，beta=1表示完全校正。 beta会从初始值线性退火到1.0。
            beta_increment_per_sampling (float): 每次采样后beta的增量。
        """
        self.tree = SumTree(self.capacity)

        # 使用一个独立的numpy数组存储实际的经验元组 (s, a, r, s', done)
        self.data = np.zeros(self.capacity, dtype=object)

        # 指针和大小
        self.write_ptr = 0
        self.size = 0
        self.state_is_array = None

    def push(self, state, action, reward, next_state, done):
        """将一个经验元组存入缓冲区。"""
        # 1. 存储经验数据
        if self.state_is_array is None:
            self.state_is_array = isinstance(state, np.ndarray)
        experience = (state, action, reward, next_state, done)
        self.data[self.write_ptr] = experience

        # 2. 在SumTree中为新经验设置最大优先级
        self.tree.add(self.max_priority)

        # 3. 更新指针和大小
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        从缓冲区中采样一个批次的经验。

        Returns:
            batch (list): 经验元组的列表。
            indices (np.array): 采样的经验在SumTree中的索引，用于后续更新优先级。
            is_weights (np.array): 对应的重要性采样权重。
        """
        batch = []
        indices = np.empty((batch_size,), dtype=np.int32)
        is_weights = np.empty((batch_size,), dtype=np.float32)

        # 1. 计算采样分段
        total_p = self.tree.total_p
        segment = total_p / batch_size

        # 2. 对beta进行退火，使其逐渐趋近1.0
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # 3. 进行分层采样
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            # 从SumTree中获取叶子节点
            (tree_idx, priority) = self.tree.get_leaf(s)

            # 计算采样概率和重要性采样权重
            sampling_prob = priority / total_p
            is_weights[i] = (self.size * sampling_prob) ** -self.beta

            # 存储索引和经验
            indices[i] = tree_idx
            # data_idx 与 tree.write_ptr 的逻辑是独立的，tree中前面还有父节点，所以需要减去capacity-1
            data_idx = tree_idx - self.capacity + 1
            batch.append(self.data[data_idx])

        # 4. 归一化权重以稳定训练
        is_weights /= is_weights.max()

        # 5. 解压批次数据
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "states": np.stack(states, axis=0)
            if self.state_is_array
            else np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.stack(next_states, axis=0)
            if self.state_is_array
            else np.array(next_states),
            "dones": np.array(dones, dtype=bool),
            "indices": indices,
            "is_weights": is_weights,
        }

    def batch_update(self, tree_indices: np.ndarray, abs_td_errors: np.ndarray):
        """
        在一次学习后，根据新的TD误差批量更新经验的优先级。

        Args:
            tree_indices (np.array): `sample`方法返回的树索引。
            abs_td_errors (np.array): 对应经验的绝对TD误差。
        """
        # 加上一个小的epsilon防止优先级为0
        priorities = (abs_td_errors + self.epsilon) ** self.alpha

        # 确保优先级不会超过设定的最大值
        clipped_priorities = np.minimum(priorities, self.max_priority)

        for idx, p in zip(tree_indices, clipped_priorities):
            self.tree.update(idx, p)

    def __len__(self):
        return self.size


@dataclass
class DQN:
    hiddens: tuple[int, ...] = (64, 64)
    bn: bool = False
    dropout: float = 0.0
    dueling_hiddens: tuple[int, ...] = ()
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
    prioritized_replay: bool = False
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment_per_sampling: float = 0.001

    def _get_n_inputs(self, space: gym.Space) -> int:
        if isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        elif isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
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
                state_tensor = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
            else:
                raise NotImplementedError
            self.online_net.eval()
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
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

    def train(self, batch: dict[str, np.ndarray]) -> tuple[float, torch.Tensor]:
        # 从缓冲区采样
        states, actions, rewards, next_states, dones = (
            batch["states"],
            batch["actions"],
            batch["rewards"],
            batch["next_states"],
            batch["dones"],
        )

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
        td_error = y_target - q_pred
        loss = td_error.pow(2).mean()

        # 4. 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        # 5. 定期更新目标网络
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.update_target_net()

        return loss.item(), td_error

    def fit(self, env: gym.Env):
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            "DQN only supports Discrete action spaces"
        )
        self._n_inputs = self._get_n_inputs(env.observation_space)
        self._n_outputs = int(env.action_space.n)

        self.online_net = DQNNet(
            self._n_inputs,
            self._n_outputs,
            self.hiddens,
            self.dueling_hiddens,
            self.bn,
            self.dropout,
        ).to(self.device)
        self.target_net = DQNNet(
            self._n_inputs,
            self._n_outputs,
            self.hiddens,
            self.dueling_hiddens,
            self.bn,
            self.dropout,
        ).to(self.device)
        self.update_target_net()
        self.target_net.eval()
        self.online_net.train()

        # 定义优化器和损失函数
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.buffer_size,
                alpha=self.per_alpha,
                beta=self.per_beta,
                beta_increment_per_sampling=self.per_beta_increment_per_sampling,
            )
        else:
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
                    loss, td_error = self.train(batch)
                    if self.prioritized_replay:
                        self.replay_buffer.batch_update(
                            batch["indices"], td_error.detach().abs().cpu().numpy()
                        )
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
    results_root = "./results/PER"
    os.makedirs(results_root, exist_ok=True)

    env = gym.make("LunarLander-v3")
    model = DQN(
        device="cuda",
        soft_update_tau=1e-3,
        n_episodes=2000,
        early_stop_target=250,
        # double_dqn=True,
        # dueling_hiddens=(64,),
        prioritized_replay=True,
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
