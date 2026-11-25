"""
Improve the predictions of already trained models, without retraining:
  - TTA (test-time augmentation): mean of the predictions over the 8 rotations / flips of each image
  - ensemble: mean of the probabilities of several models
  - temperature scaling of the softmax, chosen on the VALIDATION set only
Every combination is scored on the validation and test sets of the group split (KL divergence with the
'soft' aggregation + pixel-wise recall / precision / F1 / IoU). The best combination on the validation
set is saved in results/<name>/ with the same files as the other models of the group.

Usage (repo root, TF_USE_LEGACY_KERAS=1):
    python improve_eval.py                       # the 2 most recent experiments with a test split
    python improve_eval.py --xps XP_A XP_B       # chosen experiments (folder names in experiments/)
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from framework.dataset import LandCoverData as LCD  # noqa: E402
from framework.eval import epsilon_kl_divergence  # noqa: E402
from evaluate_test import confusion_table, load_tif, METRIC_COLS  # noqa: E402

C = LCD.N_CLASSES
EVAL_IDX = [c for c in range(C) if c not in LCD.IGNORED_CLASSES_IDX]
TEMPERATURES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
T1 = TEMPERATURES.index(1.0)
TRANSFORMS = [(k, f) for f in (False, True) for k in range(4)]   # 4 rotations x (no flip, flip)


def forward(x, k, f):
    x = np.flip(x, axis=2) if f else x
    return np.rot90(x, k, axes=(1, 2))


def backward(p, k, f):
    p = np.rot90(p, -k, axes=(1, 2))
    return np.flip(p, axis=2) if f else p


def predict_probs(model, x, tta):
    """(B, H, W, C) probabilities, averaged over the 8 dihedral transforms if tta."""
    if not tta:
        return np.asarray(model.predict_on_batch(x), dtype=np.float32)
    batch = np.concatenate([forward(x, k, f) for k, f in TRANSFORMS])
    probs = np.asarray(model.predict_on_batch(np.ascontiguousarray(batch)), dtype=np.float32)
    probs = probs.reshape(len(TRANSFORMS), len(x), *probs.shape[1:])
    return np.mean([backward(p, k, f) for p, (k, f) in zip(probs, TRANSFORMS)], axis=0)


def proportions_per_temperature(probs):
    """(B, len(TEMPERATURES), C) mean class probabilities of the images, for every temperature."""
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    out = []
    for t in TEMPERATURES:
        z = logp / t
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        out.append((e / e.sum(axis=-1, keepdims=True)).mean(axis=(1, 2)))
    return np.stack(out, axis=1)


def kl(true, pred):
    return epsilon_kl_divergence(np.array(true[:, EVAL_IDX], dtype=np.float64),
                                 np.array(pred[:, EVAL_IDX], dtype=np.float64))


def run_split(models, names, ids, dataset, batch_size):
    """Predict every image once and accumulate the results of every configuration."""
    configs = []
    for n in names:
        configs += [n, f'{n} + TTA']
    if len(models) > 1:
        configs += ['ensemble', 'ensemble + TTA']
    props = {c: [] for c in configs}
    cms = {c: np.zeros((C, C), dtype=np.int64) for c in configs}
    true = []
    for start in range(0, len(ids), batch_size):
        b = ids[start:start + batch_size]
        x = np.stack([load_tif(dataset/'train'/'images'/f'{i}.tif') for i in b]).astype(np.float32) / LCD.TRAIN_PIXELS_MAX
        y = np.stack([load_tif(dataset/'train'/'masks'/f'{i}.tif') for i in b]).astype(np.int64)
        true.append(np.stack([np.bincount(m.ravel(), minlength=C) / m.size for m in y]))
        probs = {}
        for m, n in zip(models, names):
            probs[n] = predict_probs(m, x, tta=False)
            probs[f'{n} + TTA'] = predict_probs(m, x, tta=True)
        if len(models) > 1:
            probs['ensemble'] = np.mean([probs[n] for n in names], axis=0)
            probs['ensemble + TTA'] = np.mean([probs[f'{n} + TTA'] for n in names], axis=0)
        valid = ~np.isin(y, LCD.IGNORED_CLASSES_IDX)
        for c in configs:
            props[c].append(proportions_per_temperature(probs[c]))
            pred = probs[c].argmax(-1)
            cms[c] += np.bincount(y[valid] * C + pred[valid], minlength=C * C).reshape(C, C)
        print(f"    {min(start + batch_size, len(ids))}/{len(ids)} images", end='\r', flush=True)
    print()
    return np.concatenate(true), {c: np.concatenate(v) for c, v in props.items()}, cms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/'train_config.yaml'))
    parser.add_argument('--xps', nargs='*', default=None, help="experiment folder names (default: the 2 latest)")
    parser.add_argument('--names', nargs='*', default=None,
                        help="short names of the experiments (default: scratch, imagenet)")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--tag', default='',
                        help="suffix of the output files / folder (to keep the results of previous runs)")
    parser.add_argument('--submission', action='store_true',
                        help="also predict the official challenge test set with the best configuration")
    args = parser.parse_args()
    sfx = f'_{args.tag}' if args.tag else ''

    cfg = yaml.safe_load(open(args.config))
    dataset = (ROOT/Path(cfg['dataset_folder']).expanduser()).resolve()
    xp_root = (ROOT/Path(cfg['xp_rootdir']).expanduser()).resolve()
    if args.xps:
        xps = [xp_root/x for x in args.xps]
    else:
        xps = sorted(d for d in xp_root.iterdir() if (d/'test_samples.csv').is_file() and (d/'fit_logs.csv').is_file())[-2:]
    names = args.names or (['scratch', 'imagenet'] if len(xps) == 2 else [x.name for x in xps])

    import tensorflow as tf
    models = []
    for xp, n in zip(xps, names):
        logs = pd.read_csv(xp/'fit_logs.csv')
        best = int(logs['val_kl'].idxmin()) + 1
        print(f"{n}: {xp.name}, best epoch on validation = {best}/{len(logs)}")
        models.append(tf.keras.models.load_model(str(xp/'checkpoints'/f'epoch{best}'), compile=False))
    val_ids = pd.read_csv(xps[0]/'val_samples.csv').squeeze('columns').tolist()
    test_ids = pd.read_csv(xps[0]/'test_samples.csv').squeeze('columns').tolist()
    for xp in xps[1:]:
        assert pd.read_csv(xp/'test_samples.csv').squeeze('columns').tolist() == test_ids, "different test sets"

    print("Validation:")
    val_true, val_props, val_cms = run_split(models, names, val_ids, dataset, args.batch_size)
    print("Test:")
    test_true, test_props, test_cms = run_split(models, names, test_ids, dataset, args.batch_size)

    rows = []
    for c in val_props:
        kl_val = [kl(val_true, val_props[c][:, i]) for i in range(len(TEMPERATURES))]
        i_best = int(np.argmin(kl_val))                       # temperature chosen on the validation set
        mean_test = confusion_table(test_cms[c])[METRIC_COLS].mean()
        rows.append({'configuration': c,
                     'KL_val (T=1)': kl_val[T1], 'temperature': TEMPERATURES[i_best], 'KL_val (T)': kl_val[i_best],
                     'KL_test (T=1)': kl(test_true, test_props[c][:, T1]),
                     'KL_test (T)': kl(test_true, test_props[c][:, i_best]),
                     'test Recall (%)': mean_test['Recall (%)'], 'test Precision (%)': mean_test['Precision (%)'],
                     'test F1-score (%)': mean_test['F1-score (%)'], 'test IoU (%)': mean_test['IoU (%)'],
                     '_i': i_best})
    df = pd.DataFrame(rows)
    out_dir = ROOT/'results'
    out_dir.mkdir(exist_ok=True)
    df.drop(columns='_i').round(5).to_csv(out_dir/f'summary_improvements{sfx}.csv', index=False)
    pd.set_option('display.width', 200)
    print("\n" + df.drop(columns='_i').round(4).to_string(index=False))

    # best configuration on the VALIDATION set -> same files as the other models of the group
    best = df.loc[df['KL_val (T)'].idxmin()]
    c, i = best['configuration'], int(best['_i'])
    out = out_dir/('deeplabv3plus_mobilenetv2_best_' + c.replace(' + ', '_').replace(' ', '_') + sfx)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(test_props[c][:, i], index=pd.Index(test_ids, name='sample_id'), columns=LCD.CLASSES)\
        .to_csv(out/'best_final_test_predicted.csv')
    df_cm = confusion_table(test_cms[c])
    df_cm.to_csv(out/'confusion_best_test.csv')
    df_cm[METRIC_COLS].mean().rename('mean').rename_axis('metric').to_csv(out/'mean_metrics_best_test.csv')
    pd.DataFrame([{'configuration': c, 'temperature': TEMPERATURES[i], 'KL_soft': best['KL_test (T)'],
                   'experiments': ' + '.join(x.name for x in xps)}]).to_csv(out/'eval_best_test_results.csv', index=False)
    with open(out/'config.yaml', 'w') as f:
        yaml.safe_dump({'configuration': c, 'temperature': TEMPERATURES[i], 'tta_transforms': 8 if 'TTA' in c else 1,
                        'experiments': [x.name for x in xps], 'names': names}, f, sort_keys=False)
    print(f"\nBest configuration on the validation set: {c}, temperature {TEMPERATURES[i]}")
    print(f"KL divergence (test): {best['KL_test (T)']}")
    if args.submission:
        # official challenge test set, same configuration (models, TTA, temperature) as chosen on validation
        test_images = sorted((dataset/'test'/'images').glob('*.tif'))
        if not test_images:
            print("No official test images found, no submission file")
        else:
            use = [k for k, n in enumerate(names) if c in (n, f'{n} + TTA')] or list(range(len(models)))
            tta = c.endswith('TTA')
            t = TEMPERATURES[i]
            rows_sub = []
            for start in range(0, len(test_images), args.batch_size):
                x = np.stack([load_tif(f) for f in test_images[start:start + args.batch_size]])
                x = x.astype(np.float32) / LCD.TRAIN_PIXELS_MAX
                p = np.mean([predict_probs(models[k], x, tta) for k in use], axis=0)
                z = np.log(np.clip(p, 1e-12, 1.0)) / t
                z -= z.max(axis=-1, keepdims=True)
                e = np.exp(z)
                rows_sub.append((e / e.sum(axis=-1, keepdims=True)).mean(axis=(1, 2)))
            pd.DataFrame(np.concatenate(rows_sub), index=pd.Index([int(f.stem) for f in test_images], name='sample_id'),
                         columns=LCD.CLASSES).to_csv(out/'submission.csv')
            print(f"Challenge submission ({len(test_images)} images): {out/'submission.csv'}")
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
