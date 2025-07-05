import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    fns = {
        "DQN": "./results/DQN/history.csv",
        "Double DQN": "./results/doubleDQN/history.csv",
        "Dueling DQN": "./results/duelingDQN/history.csv",
        "PER DQN": "./results/PER/history.csv",
    }

    df = {}
    for name, fn in fns.items():
        dfi = pd.read_csv(fn, index_col=0)
        df[name] = dfi

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    palette = sns.color_palette()
    all_handles, all_labels = [], []
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
            axs[i].set_title(key)
            axs[i].set_xlabel("Episode")

            axs[i].legend()
            handles, labels = axs[i].get_legend_handles_labels()
            axs[i].legend_.remove()
            for hi, li in zip(handles, labels):
                if li not in all_labels:
                    all_handles.append(hi)
                    all_labels.append(li)
    fig.legend(all_handles, all_labels, loc="outside lower center", ncols=len(fns))
    # fig.tight_layout()
    fig.savefig("./results/history.png")
