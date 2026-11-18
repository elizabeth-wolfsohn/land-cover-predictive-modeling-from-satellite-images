import pandas as pd
import matplotlib.pyplot as plt

max_iter = 50
if __name__ == "__main__":
    exp_path = "challenge-ens/code/experiments/18-11-2026_00:55:48"
    df = pd.read_csv(exp_path + "/fit_logs.csv")
    df = df[:max_iter]
    ax = df.plot(x="epoch", y=["loss", "val_loss"], marker="o")
    ax.set_ylabel("loss"); ax.grid(True)
    plt.show()