"""
Make the figures without training:
  - samples of the dataset                  python -m framework.visualize --dataset-only
  - predictions of a trained model          python -m framework.visualize --epoch 20 --num 6
  - training curves of an experiment        (made automatically when an experiment is given)

The dataset folder, experiments folder and experiment name are read from infer_config.yaml
(can be overridden with the command-line options).
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import argparse
import random
import yaml
import numpy as np
import pandas as pd

from framework.visualization import (
    load_sample, plot_dataset_samples, plot_predictions, plot_training_curves, open_file
)
from framework.dataset import LandCoverData as LCD


def _parse_args():
    parser = argparse.ArgumentParser('Visualization script')
    parser.add_argument('--config', '-c', default=str(ROOT_DIR/'infer_config.yaml'),
                        help="YAML config giving dataset_folder, xp_rootdir, xp_name, checkpoint_epoch")
    parser.add_argument('--dataset-only', action='store_true', help="only plot dataset samples")
    parser.add_argument('--xp', default=None, help="experiment name (default: xp_name of the config)")
    parser.add_argument('--epoch', type=int, default=None,
                        help="checkpoint epoch (default: checkpoint_epoch of the config, or the last one)")
    parser.add_argument('--set', choices=['val', 'test_split', 'test'], default='val',
                        help="images to predict on: 'val' (with ground truth) or 'test'")
    parser.add_argument('--num', type=int, default=None,
                        help="number of images to plot (default: 10 for the dataset, 6 for predictions)")
    parser.add_argument('--seed', type=int, default=0, help="random seed to pick the images")
    parser.add_argument('--no-open', action='store_true', help="do not open the figures")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    args.dataset_folder = (ROOT_DIR/Path(cfg['dataset_folder']).expanduser()).resolve()
    args.xp_rootdir = (ROOT_DIR/Path(cfg['xp_rootdir']).expanduser()).resolve()
    args.xp = args.xp or cfg.get('xp_name', 'last')
    if args.epoch is None:
        args.epoch = cfg.get('checkpoint_epoch')
    assert args.dataset_folder.is_dir(), f"dataset_folder not found: {args.dataset_folder}"
    return args


def main():
    args = _parse_args()
    rng = random.Random(args.seed)

    if args.dataset_only:
        out_dir = ROOT_DIR/'figures'
        out_dir.mkdir(exist_ok=True)
        files = sorted(args.dataset_folder.glob('train/images/*.tif'))
        files = rng.sample(files, min(args.num or 10, len(files)))
        png = plot_dataset_samples(files, out_dir/'dataset_samples.png', num=len(files))
        print(f"Saved {png}")
        if not args.no_open:
            open_file(png)
        return

    # experiment directory
    xps = [d for d in args.xp_rootdir.iterdir() if d.is_dir()] if args.xp_rootdir.is_dir() else []
    if not xps:
        sys.exit(f"Aucune experience trouvee dans {args.xp_rootdir}.\n"
                 "Lance d'abord un entrainement (python -m framework.train), "
                 "ou utilise --dataset-only pour visualiser uniquement le dataset.")
    if args.xp == 'last':
        xp_dir = max(xps, key=lambda d: d.stat().st_mtime)
    else:
        xp_dir = args.xp_rootdir/args.xp
    assert xp_dir.is_dir(), f"experiment not found: {xp_dir}"
    plots_dir = xp_dir/'plots'
    plots_dir.mkdir(exist_ok=True)
    print(f"Experiment: {xp_dir}")
    to_open = []

    # training curves from the CSV logs
    if (xp_dir/'fit_logs.csv').is_file():
        logs = pd.read_csv(xp_dir/'fit_logs.csv')
        to_open.append(plot_training_curves(logs, plots_dir/'training_curves.png'))
        print(f"Saved {to_open[-1]}")

    # checkpoint to use
    ckpt_dir = xp_dir/'checkpoints'
    epochs = sorted(int(d.name[len('epoch'):]) for d in ckpt_dir.glob('epoch*'))
    assert epochs, f"no checkpoint in {ckpt_dir}"
    epoch = args.epoch if args.epoch in epochs else epochs[-1]
    if args.epoch is not None and args.epoch not in epochs:
        print(f"Checkpoint epoch {args.epoch} not found, using the last one ({epoch})")

    # images to predict on
    if args.set in ('val', 'test_split'):
        csv_name = 'val_samples.csv' if args.set == 'val' else 'test_samples.csv'
        ids = pd.read_csv(xp_dir/csv_name).squeeze('columns').tolist()
        files = [args.dataset_folder/'train/images'/f'{i}.tif' for i in ids]
    else:
        files = sorted(args.dataset_folder.glob('test/images/*.tif'))
    files = rng.sample(files, min(args.num or 6, len(files)))
    images, masks = zip(*(load_sample(f) for f in files))
    images = np.stack(images).astype(np.float32) / LCD.TRAIN_PIXELS_MAX
    masks = np.stack(masks) if args.set != 'test' else None

    import tensorflow as tf
    model = tf.keras.models.load_model(str(ckpt_dir/f'epoch{epoch}'), compile=False)
    probs = model.predict(images, verbose=0)
    png = plot_predictions(images, masks, probs,
                           plots_dir/f'predictions_{args.set}_epoch{epoch}_seed{args.seed}.png',
                           title=f'{xp_dir.name} - époque {epoch} - images {args.set}',
                           sample_ids=[f.stem for f in files])
    to_open.append(png)
    print(f"Saved {png}")
    if not args.no_open:
        for p in to_open:
            open_file(p)


if __name__ == '__main__':
    main()
