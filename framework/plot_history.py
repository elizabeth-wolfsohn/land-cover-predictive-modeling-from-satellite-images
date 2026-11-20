import pandas as pd
import matplotlib.pyplot as plt

max_iter = 90
if __name__ == "__main__":
    exp_path = "challenge-ens/code/experiments/19-11-2026_17:41:48"
    df = pd.read_csv(exp_path + "/fit_logs.csv")
    df = df[:max_iter]
    ax = df.plot(x="epoch", y=["loss", "val_loss"], marker="o")
    ax.set_ylabel("loss"); ax.grid(True)
    ax.set_ylim(0,1)
    plt.show()

