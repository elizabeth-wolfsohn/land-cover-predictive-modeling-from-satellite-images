"""
Plot the training curves (fit_logs.csv) of one or several experiments, overlaid for comparison.
Usage: python -m framework.plot_logs code/experiments/<xp1> [code/experiments/<xp2> ...] [-o curves.png]
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BENCHMARK_KL = 0.0593

parser = argparse.ArgumentParser('Plot training curves')
parser.add_argument('xp_dirs', nargs='+', type=Path, help="experiment directories (containing fit_logs.csv)")
parser.add_argument('--out', '-o', type=Path, default=Path('curves.png'), help="output image file")
args = parser.parse_args()

logs = {d.name: pd.read_csv(d/'fit_logs.csv') for d in args.xp_dirs}
# accuracy/mIoU panel only if logged (added after the first runs)
has_acc = any('val_acc' in df for df in logs.values())
fig, axs = plt.subplots(1, 3 if has_acc else 2, figsize=(17 if has_acc else 12, 4.5))

for (name, df), color in zip(logs.items(), plt.rcParams['axes.prop_cycle'].by_key()['color']):
    axs[0].plot(df.epoch, df.loss, color=color, ls='--', label=f'{name} train')
    axs[0].plot(df.epoch, df.val_loss, color=color, label=f'{name} val')
    axs[1].plot(df.epoch, df.val_kl_argmax, color=color, ls='--', label=f'{name} argmax')
    axs[1].plot(df.epoch, df.val_kl_soft, color=color, label=f'{name} soft')
    if 'val_acc' in df:
        axs[2].plot(df.epoch, df.acc, color=color, ls=':', label=f'{name} acc train')
        axs[2].plot(df.epoch, df.val_acc, color=color, label=f'{name} acc val')
        axs[2].plot(df.epoch, df.val_miou, color=color, ls='--', label=f'{name} mIoU val')

axs[0].set(title='Cross-entropy', xlabel='epoch', ylabel='loss')
axs[1].axhline(BENCHMARK_KL, color='grey', ls=':', label='benchmark U-Net')
axs[1].set(title='KL divergence (validation)', xlabel='epoch', ylabel='KL', yscale='log')
if has_acc:
    axs[2].set(title='Accuracy / mIoU', xlabel='epoch')
for ax in axs:
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f"Saved {args.out}")
