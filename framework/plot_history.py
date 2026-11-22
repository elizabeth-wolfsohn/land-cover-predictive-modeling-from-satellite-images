import pandas as pd
import matplotlib.pyplot as plt

max_iter = 200
if __name__ == "__main__":
    exp_path = "challenge-ens/code/experiments/22-11-2026_11:04:49"
    df = pd.read_csv(exp_path + "/fit_logs.csv")
    df = df[:max_iter]
    ax = df.plot(x="epoch", y=["loss", "val_loss"], marker="o")
    ax.set_ylabel("loss"); ax.grid(True)
    ax.set_ylim(0,1)
    plt.show()
    ax = df.plot(x="epoch", y=["kl", "val_kl"], marker="o")
    ax.set_ylabel("kl"); ax.grid(True)
    ax.set_ylim(0,0.5)
    plt.show()

