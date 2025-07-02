import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    fns = {
        "DQN": "./results/DQN/history.csv",
        "Double DQN": "./results/doubleDQN/history.csv",
    }

    df = {}
    for name, fn in fns.items():
        dfi = pd.read_csv(fn, index_col=0)
        df[name] = dfi

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    palette = sns.color_palette()
    for i, key in enumerate(["loss", "total_reward"]):
        for j, (name, dfi) in enumerate(df.items()):
            axs[i].plot(
                dfi[key],
                label=f"{name}(True values)",
                alpha=0.3,
                color=palette[j],
            )
            axs[i].plot(
                dfi[key].rolling(100).mean(),
                label=f"{name}(100-episode average)",
                alpha=1.0,
                color=palette[j],
            )
            axs[i].legend()
            axs[i].set_title(key)
            axs[i].set_xlabel("Episode")
    fig.tight_layout()
    fig.savefig("./results/history.png")
