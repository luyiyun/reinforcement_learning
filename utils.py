import numpy as np
import matplotlib.pyplot as plt


def plot_rewards_history(
    rewards_historys: dict[str, np.ndarray], window_size: int = 500
) -> None:
    fig, ax = plt.subplots()
    for key, rewards_history in rewards_historys.items():
        moving_avg = np.convolve(
            rewards_history, np.ones(window_size) / window_size, mode="valid"
        )
        ax.plot(
            np.arange(window_size - 1, len(rewards_history)),
            moving_avg,
            label=key,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    return fig, ax


def plot_reward_history(
    rewards_history: np.ndarray,
    window_size: int = 500,
):
    fig, ax = plt.subplots()
    moving_avg = np.convolve(
        rewards_history, np.ones(window_size) / window_size, mode="valid"
    )
    ax.plot(
        np.arange(window_size - 1, len(rewards_history)),
        moving_avg,
        label=f"Moving Average Reward ({window_size})",
    )
    ax.plot(rewards_history, label="True Reward", alpha=0.3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    return fig, ax


def plot_blackjack_value_function(V: np.ndarray, save_fn: str) -> None:
    vmin, vmax = V.min(), V.max()
    fig, axs = plt.subplots(ncols=3, figsize=(10, 6), width_ratios=(0.49, 0.49, 0.02))
    for i in range(2):
        ax = axs[i]
        Vi = V[..., i]
        im = ax.imshow(Vi, vmin=vmin, vmax=vmax, cmap="RdYlGn")
        ax.set_xticks(
            range(Vi.shape[1]),
            labels=["A"] + list(range(1, 11)),
        )
        ax.set_yticks(range(Vi.shape[0]), labels=range(1, 33))
        ax.set_xlabel("Dealer Card")
        ax.set_ylabel("Player Sum")
        ax.set_title(f"Usable Ace = {i}")

    fig.colorbar(im, cax=axs[2])

    fig.tight_layout()
    fig.savefig(save_fn)


def plot_frozen_lake_value_function(V: np.ndarray, save_fn: str) -> None:
    fig, axs = plt.subplots(ncols=2, figsize=(6, 6), width_ratios=(0.95, 0.05))
    im = axs[0].imshow(V.reshape(4, 4), cmap="RdYlGn")
    fig.colorbar(im, cax=axs[1])
    fig.tight_layout()
    fig.savefig(save_fn)
