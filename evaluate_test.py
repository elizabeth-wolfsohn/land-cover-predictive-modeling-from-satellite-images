"""
Evaluate every trained experiment on the validation and held-out test sets of the group split:
best epoch on validation (lowest val_kl), KL divergence (argmax / soft) and pixel-wise confusion
matrix with recall, precision, F1-score and IoU per class (same definitions as the group's eval.py).
Usage: python evaluate_test.py [--submission] [--save-masks]
"""
import sys
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from framework.dataset import LandCoverData as LCD  # noqa: E402
from framework.eval import epsilon_kl_divergence  # noqa: E402

IGNORED = [LCD.CLASSES[i] for i in LCD.IGNORED_CLASSES_IDX]
METRIC_COLS = ["Recall (%)", "Precision (%)", "F1-score (%)", "IoU (%)"]


def load_tif(path):
    from tifffile import imread
    return imread(str(path))


def predict(model, image_files, mask_files=None, batch_size=32, save_mask_dir=None):
    C = LCD.N_CLASSES
    soft, hard, true = [], [], []
    cm = np.zeros((C, C), dtype=np.int64)
    if save_mask_dir is not None:
        from tifffile import imwrite
        save_mask_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(image_files), batch_size):
        files = image_files[start:start + batch_size]
        x = np.stack([load_tif(f) for f in files]).astype(np.float32) / LCD.TRAIN_PIXELS_MAX
        probs = np.asarray(model.predict_on_batch(x), dtype=np.float64)
        pred = probs.argmax(-1)
        soft.append(probs.mean(axis=(1, 2)))
        hard.append(np.stack([np.bincount(p.ravel(), minlength=C) / p.size for p in pred]))
        if save_mask_dir is not None:
            for f, p in zip(files, pred):
                imwrite(str(save_mask_dir/Path(f).name), p.astype(np.uint8))
        if mask_files is not None:
            for p, mf in zip(pred, mask_files[start:start + batch_size]):
                y = load_tif(mf).ravel().astype(np.int64)
                true.append(np.bincount(y, minlength=C) / y.size)
                valid = ~np.isin(y, LCD.IGNORED_CLASSES_IDX)
                cm += np.bincount(y[valid] * C + p.ravel()[valid], minlength=C * C).reshape(C, C)
    out = {'soft': np.concatenate(soft), 'argmax': np.concatenate(hard)}
    if mask_files is not None:
        out['true'] = np.stack(true)
        out['cm'] = cm
    return out


def confusion_table(cm):
    tp = np.diag(cm).astype(float)
    total_true, total_pred = cm.sum(axis=1), cm.sum(axis=0)
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    recall, precision = div(tp, total_true), div(tp, total_pred)
    f1 = div(2 * precision * recall, precision + recall)
    iou = div(tp, total_true + total_pred - tp)
    df = pd.DataFrame(cm, index=LCD.CLASSES, columns=LCD.CLASSES).iloc[2:, :]
    df.index.name = "Ground Truth Class"
    df["Recall (%)"], df["Precision (%)"] = recall[2:] * 100, precision[2:] * 100
    df["F1-score (%)"], df["IoU (%)"] = f1[2:] * 100, iou[2:] * 100
    return df


def kl(df_true, df_pred):
    return epsilon_kl_divergence(df_true.drop(columns=IGNORED).to_numpy(dtype=np.float64, copy=True),
                                 df_pred.drop(columns=IGNORED).to_numpy(dtype=np.float64, copy=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'train_config.yaml'))
    parser.add_argument('--submission', action='store_true')
    parser.add_argument('--save-masks', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dataset = (ROOT/Path(cfg['dataset_folder']).expanduser()).resolve()
    xp_root = (ROOT/Path(cfg['xp_rootdir']).expanduser()).resolve()
    assert dataset.is_dir(), f"dataset_folder not found: {dataset}"

    import tensorflow as tf

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
        ckpt = xp/'checkpoints'/f'epoch{best}'
        if not ckpt.exists():
            print(f"skip {xp.name} (checkpoint of epoch {best} missing)")
            continue
        print(f"\n=== {xp.name}: best epoch on validation = {best}/{len(logs)}")
        model = tf.keras.models.load_model(str(ckpt), compile=False)

        row = {'experiment': xp.name, 'best_epoch': best, 'epochs': len(logs)}
        for split, csv_name in (('val', 'val_samples.csv'), ('test', 'test_samples.csv')):
            ids = pd.read_csv(xp/csv_name).squeeze('columns').tolist()
            images = [dataset/'train'/'images'/f'{i}.tif' for i in ids]
            masks = [dataset/'train'/'masks'/f'{i}.tif' for i in ids]
            res = predict(model, images, masks, args.batch_size,
                          save_mask_dir=xp/f'best_{split}_pred_masks' if args.save_masks else None)
            index = pd.Index(ids, name='sample_id')
            df_true = pd.DataFrame(res['true'], index=index, columns=LCD.CLASSES)
            df_true.to_csv(xp/f'y_{split}.csv')
            for agg in ('argmax', 'soft'):
                df_pred = pd.DataFrame(res[agg], index=index, columns=LCD.CLASSES)
                df_pred.to_csv(xp/f'epoch{best}_{split}_{agg}_predicted.csv')
                row[f'KL_{split}_{agg}'] = round(kl(df_true, df_pred), 5)
            if args.save_masks:
                shutil.copy(xp/f'epoch{best}_{split}_soft_predicted.csv', xp/f'best_{split}_predicted.csv')
            df_cm = confusion_table(res['cm'])
            df_cm.to_csv(xp/f'confusion_best_{split}.csv')
            mean = df_cm[METRIC_COLS].mean().rename('mean').rename_axis('metric')
            mean.to_csv(xp/f'mean_metrics_best_{split}.csv')
            pd.DataFrame([{'split': split, 'best_epoch': best, 'KL_soft': row[f'KL_{split}_soft'],
                           'KL_argmax': row[f'KL_{split}_argmax']}])\
                .to_csv(xp/f'eval_best_{split}_results.csv', index=False)
            for col in METRIC_COLS:
                row[f'{split}_mean_{col.split(" ")[0]}'] = round(mean[col], 2)
            print(f"    {split:4s}: KL argmax = {row[f'KL_{split}_argmax']:.5f} | KL soft = {row[f'KL_{split}_soft']:.5f}"
                  f" | mean F1 = {mean['F1-score (%)']:.1f} % | mean IoU = {mean['IoU (%)']:.1f} %")
            if split == 'test':
                print(df_cm[METRIC_COLS].round(1).to_string())

        if args.submission:
            test_images = sorted((dataset/'test'/'images').glob('*.tif'))
            if test_images:
                res = predict(model, test_images, batch_size=args.batch_size)
                pd.DataFrame(res['soft'], index=pd.Index([int(f.stem) for f in test_images], name='sample_id'),
                             columns=LCD.CLASSES).to_csv(xp/'submission.csv')
            else:
                print("    no official test images found, no submission file")
        rows.append(row)
        del model
        tf.keras.backend.clear_session()

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
