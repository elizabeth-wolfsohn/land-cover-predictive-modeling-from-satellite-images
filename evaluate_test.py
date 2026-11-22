"""
Evaluate every trained experiment on the held-out test split of the group.

For each experiment of experiments/ trained with the predefined split (test_samples.csv present):
  1. pick the best epoch on the VALIDATION set (lowest val_kl), never on the test set
  2. predict the validation and test images with the 'argmax' and 'soft' aggregations
  3. compute the challenge metric (KL divergence) against the ground truth
  4. optionally predict the official challenge test set (CSV to submit on the website)
  5. copy the small result files (logs, curves, predictions, CSV, no checkpoints) to results/<experiment>/

A summary table is printed and saved to results/summary_test_split.csv.

Usage (from the repo root, after the trainings):
    python evaluate_test.py
    python evaluate_test.py --submission       # also predicts the official test set
"""
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run(cmd):
    r = subprocess.run([sys.executable, '-m'] + cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return r.stdout


def kl_score(gt_csv, pred_csv):
    out = run(['framework.eval', '-g', str(gt_csv), '-p', str(pred_csv)])
    return float(out.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'train_config.yaml'),
                        help="train config giving dataset_folder, xp_rootdir and the split CSV files")
    parser.add_argument('--submission', action='store_true',
                        help="also predict the official challenge test set (soft aggregation)")
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dataset = (ROOT/Path(cfg['dataset_folder']).expanduser()).resolve()
    xp_root = (ROOT/Path(cfg['xp_rootdir']).expanduser()).resolve()
    test_labels = (ROOT/Path(cfg.get('test_image_path') or 'local_dataset/csv_files/test_images.csv')
                   .expanduser()).resolve()
    test_labels = test_labels.with_name(test_labels.name.replace('_images', '_labels'))
    assert test_labels.is_file(), f"test labels not found: {test_labels}"
    assert dataset.is_dir(), f"dataset_folder not found: {dataset}"

    from framework.dataset import LandCoverData as LCD
    from tifffile import imread

    rows = []
    for xp in sorted(d for d in xp_root.iterdir() if d.is_dir()):
        logs_csv = xp/'fit_logs.csv'
        if not (xp/'test_samples.csv').is_file() or not logs_csv.is_file():
            print(f"skip {xp.name} (no predefined test split or no logs)")
            continue
        logs = pd.read_csv(logs_csv)
        if 'val_kl' not in logs or logs.empty:
            print(f"skip {xp.name} (no val_kl in the logs)")
            continue
        best = int(logs['val_kl'].idxmin()) + 1
        if not (xp/'checkpoints'/f'epoch{best}').exists():
            print(f"skip {xp.name} (checkpoint of epoch {best} missing)")
            continue
        print(f"\n=== {xp.name}: best epoch on validation = {best}/{len(logs)}")

        # ground truth of the validation set (from the masks)
        val_ids = pd.read_csv(xp/'val_samples.csv').squeeze('columns').tolist()
        y_val = [np.bincount(imread(dataset/'train'/'masks'/f'{i}.tif').ravel(), minlength=LCD.N_CLASSES)
                 / (LCD.IMG_SIZE * LCD.IMG_SIZE) for i in val_ids]
        pd.DataFrame(y_val, index=pd.Index(val_ids, name='sample_id'), columns=LCD.CLASSES)\
            .to_csv(xp/'y_val.csv')

        row = {'experiment': xp.name, 'best_epoch': best, 'epochs': len(logs)}
        splits = [('val', xp/'y_val.csv'), ('test_split', test_labels)]
        for split, gt in splits:
            for agg in ('argmax', 'soft'):
                infer_cfg = dict(seed=42, batch_size=args.batch_size, dataset_folder=str(dataset),
                                 xp_rootdir=str(xp_root), xp_name=xp.name, set=split,
                                 checkpoint_epoch=best, aggregation=agg)
                tmp = xp/'tmp_infer.yaml'
                yaml.safe_dump(infer_cfg, open(tmp, 'w'))
                run(['framework.infer', '-c', str(tmp)])
                score = kl_score(gt, xp/f'epoch{best}_{split}_{agg}_predicted.csv')
                name = 'val' if split == 'val' else 'test'
                row[f'KL_{name}_{agg}'] = round(score, 5)
                print(f"    KL {name:4s} ({agg:6s}) = {score:.5f}")
        if args.submission:
            infer_cfg.update(set='test', aggregation='soft')
            yaml.safe_dump(infer_cfg, open(tmp, 'w'))
            run(['framework.infer', '-c', str(tmp)])
            shutil.copy(xp/f'epoch{best}_test_soft_predicted.csv', xp/'submission.csv')
        tmp.unlink(missing_ok=True)
        rows.append(row)

        # small result files -> results/<experiment>/ (no checkpoints)
        out = ROOT/'results'/xp.name
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        for f in xp.iterdir():
            if f.is_file():
                shutil.copy(f, out)
        if (xp/'plots').is_dir():
            shutil.copytree(xp/'plots', out/'plots')

    if not rows:
        print("No experiment trained with the predefined split was found in", xp_root)
        return
    summary = pd.DataFrame(rows)
    (ROOT/'results').mkdir(exist_ok=True)
    summary.to_csv(ROOT/'results'/'summary_test_split.csv', index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nSaved to {ROOT/'results'/'summary_test_split.csv'}")


if __name__ == '__main__':
    main()
