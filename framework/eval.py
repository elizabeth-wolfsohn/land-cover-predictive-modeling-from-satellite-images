from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from framework.dataset import LandCoverData as LCD

def epsilon_kl_divergence(y_true, y_pred):
    """
        The metric of the challenge, the Kullback-Leibler divergence between two discrete distributions.
        Modification includes adding a small constant epsilon (1e-7) to y_true and y_pred
    Args:
            y_true (numpy.array[float]): ground-truth
            y_pred (numpy.arra[float]): prediction
    Returns:
        (float): the score (lower is better)
    """
    # np.array (et non np.asarray) : copie, pour ne pas modifier les données de l'appelant
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)

    # Normalize to sum to 1 if it's not already
    y_true /= y_true.sum(1, keepdims=True)
    y_pred /= y_pred.sum(1, keepdims=True)
    # add a small constant for smoothness around 0
    y_true += 1e-7
    y_pred += 1e-7
    score = np.mean(np.sum(y_true * np.log(y_true / y_pred), 1))
    try:
        assert np.isfinite(score)
    except AssertionError as e:
        raise ValueError('score is NaN or infinite') from e
    return score


# ==========================================================================
# Métriques supplémentaires, calculées sur les vecteurs de proportions.
#
# Ce script ne voit que des proportions par image (pas les pixels) : les
# métriques ci-dessous en sont les équivalents.
#   - "soft" : précision / rappel / F1 / IoU sur la masse de proportion commune
#     min(vrai, prédit). Ce sont des bornes hautes des métriques au pixel :
#     un modèle ne peut pas faire mieux au pixel que ce que ses proportions permettent.
#   - "presence" : la classe est-elle présente dans l'image (proportion > seuil) ?
#     Précision / rappel / F1 de cette détection, et matrice de confusion 2x2 par classe.
# Pour les vraies métriques au pixel : framework/evaluate_metrics.py
# ==========================================================================
EPS = 1e-7   # même lissage que epsilon_kl_divergence


def _normalize(values):
    values = np.array(values, dtype=np.float64)
    return values / values.sum(1, keepdims=True)


def _ratio(num, den):
    """Division où un dénominateur nul donne NaN."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def per_image_scores(y_true, y_pred):
    """KL et entropie croisée de chaque image, et contribution de chaque classe à la KL.

    Mêmes normalisation et epsilon que epsilon_kl_divergence :
    la moyenne de 'kl' est exactement le score, et la somme des contributions par classe aussi.
    """
    t = _normalize(y_true) + EPS
    p = _normalize(y_pred) + EPS
    kl_contrib = t * np.log(t / p)                     # (N, C)
    cross_entropy = -np.sum(t * np.log(p), axis=1)     # = KL + entropie des vraies proportions
    return kl_contrib.sum(1), cross_entropy, kl_contrib


def per_class_metrics(y_true, y_pred, kl_contrib, class_names, threshold):
    t, p = _normalize(y_true), _normalize(y_pred)
    overlap = np.minimum(t, p).sum(0)
    union = np.maximum(t, p).sum(0)
    sum_true, sum_pred = t.sum(0), p.sum(0)

    # détection de présence de chaque classe dans chaque image
    true_present, pred_present = t > threshold, p > threshold
    tp = (true_present & pred_present).sum(0)
    fn = (true_present & ~pred_present).sum(0)
    fp = (~true_present & pred_present).sum(0)
    tn = (~true_present & ~pred_present).sum(0)
    presence_defined = (tp + fp + fn) > 0              # sinon classe jamais présente ni prédite

    def zero_if_undefined(num, den):
        # convention scikit-learn : 0 si indéfini, NaN si la classe n'est jamais en jeu
        values = np.divide(num, den, out=np.zeros(len(num)), where=den > 0)
        values[~presence_defined] = np.nan
        return values

    metrics = pd.DataFrame({
        'mean_true': t.mean(0),
        'mean_pred': p.mean(0),
        'bias': p.mean(0) - t.mean(0),                 # > 0 : classe surestimée en moyenne
        'mae': np.abs(p - t).mean(0),
        'rmse': np.sqrt(((p - t) ** 2).mean(0)),
        'kl_contribution': kl_contrib.mean(0),         # somme sur les classes = score KL
        'soft_precision': _ratio(overlap, sum_pred),
        'soft_recall': _ratio(overlap, sum_true),
        'soft_f1': _ratio(2 * overlap, sum_true + sum_pred),
        'soft_iou': _ratio(overlap, union),
        'presence_precision': zero_if_undefined(tp, tp + fp),
        'presence_recall': zero_if_undefined(tp, tp + fn),
        'presence_f1': zero_if_undefined(2 * tp, 2 * tp + fp + fn),
        'n_images_present': true_present.sum(0),
    }, index=pd.Index(class_names, name='classe'))
    confusion = pd.DataFrame({'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn},
                             index=pd.Index(class_names, name='classe'))
    return metrics, confusion


def summary_metrics(per_class, kl, cross_entropy, y_true, y_pred):
    t, p = _normalize(y_true), _normalize(y_pred)
    weights = per_class['mean_true'].to_numpy()

    def weighted(column):
        values = per_class[column].to_numpy(dtype=np.float64)
        ok = ~np.isnan(values)
        return float(np.sum(values[ok] * weights[ok]) / weights[ok].sum())

    summary = {
        'n_images': len(t),
        'kl': float(kl.mean()),
        'cross_entropy': float(cross_entropy.mean()),
        'entropy_true': float((cross_entropy - kl).mean()),            # partie incompressible de la CE
        # part des proportions correctement placée (= 1 - distance de variation totale) :
        # borne haute de l'exactitude au pixel
        'proportion_overlap': float(np.minimum(t, p).sum(1).mean()),
        'mae': float(np.abs(p - t).mean()),
    }
    for column in ('soft_precision', 'soft_recall', 'soft_f1', 'soft_iou',
                   'presence_precision', 'presence_recall', 'presence_f1'):
        summary[f'macro_{column}'] = float(per_class[column].mean())
    for column in ('soft_precision', 'soft_recall', 'soft_f1', 'soft_iou'):
        summary[f'weighted_{column}'] = weighted(column)
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in summary.items()}


def plot_presence_confusion(confusion, per_class, threshold, path):
    """Une matrice 2x2 par classe : la classe est-elle présente dans l'image, en vrai et en prédit."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = ['#f3f7fc', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
    cmap = LinearSegmentedColormap.from_list('sequential_blue', ramp)
    ink, ink_light, ink_muted = '#1f1f1f', '#ffffff', '#6b6b6b'

    names = list(confusion.index)
    n_cols = min(4, len(names))
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.0 * n_rows + 0.6), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')

    for ax, name in zip(axes.ravel(), names):
        c = confusion.loc[name]
        counts = np.array([[c['TP'], c['FN']], [c['FP'], c['TN']]], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            share = counts / counts.sum(1, keepdims=True)                # normalisé par ligne
        ax.imshow(np.nan_to_num(share) * 100, cmap=cmap, vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                if np.isnan(share[i, j]):
                    text, color = '–', ink_muted
                else:
                    text = f'{int(counts[i, j])}\n{share[i, j] * 100:.0f} %'
                    color = ink_light if share[i, j] >= 0.5 else ink
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
        f1 = per_class.loc[name, 'presence_f1']
        ax.set_title(f"{name}  (F1 {f1:.2f})" if not np.isnan(f1) else name,
                     loc='left', fontsize=10, color=ink)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['prédit\nprésent', 'prédit\nabsent'], fontsize=8, color=ink)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['réel\nprésent', 'réel\nabsent'], fontsize=8, color=ink)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"Détection de présence par classe (seuil {threshold:.0%} de l'image) — "
                 "nombre d'images et % de la ligne", x=0.01, ha='left', fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser('Evaluation script')
    parser.add_argument('--gt-file', '-g', type=str, help="ground truth CSV file")
    parser.add_argument('--pred-file', '-p', type=str, required=True, help="prediction CSV file")
    parser.add_argument('--out-csv', '-o', type=str, default=None, help="output CSV file, optional")
    parser.add_argument('--presence-threshold', type=float, default=0.01,
                        help="proportion à partir de laquelle une classe est considérée présente dans l'image")
    cli_args = parser.parse_args()
    cli_args.gt_file = Path(cli_args.gt_file).resolve()
    assert cli_args.gt_file.is_file()
    cli_args.pred_file = Path(cli_args.pred_file).resolve()
    assert cli_args.pred_file.is_file()
    return cli_args

if __name__ == '__main__':

    args = _parse_args()
    df_y_true = pd.read_csv(args.gt_file, index_col=0, sep=',')
    df_y_pred = pd.read_csv(args.pred_file, index_col=0, sep=',')

    if not len(df_y_true.index.intersection(df_y_pred.index)) == len(df_y_pred.index):
        raise ValueError("some samples IDs in y_pred are not present in y_true")

    # select subset of labels corresponding to predicted samples (train or val)
    df_y_true = df_y_true.loc[df_y_pred.index]

    # remove columns of the ignored classes if present
    ignored_classes = [LCD.CLASSES[i] for i in LCD.IGNORED_CLASSES_IDX]
    df_y_pred = df_y_pred.drop(ignored_classes, axis=1, errors='ignore')
    df_y_true = df_y_true.drop(ignored_classes, axis=1, errors='ignore')

    # use same order for columns
    df_y_pred = df_y_pred.loc[:, df_y_true.columns]

    if not df_y_pred.shape == df_y_true.shape:
        raise ValueError(f"shapes not equal between y_true and y_pred")

    if np.any(np.allclose(df_y_pred.sum(1), 0.)):
        # if any row predicted sums to 0, raise an error
        raise ValueError("some row vector(s) in y_pred sum to 0")

    # une image sans pixel évalué (que du no_data / des nuages) n'a pas de vraie distribution :
    # elle rendrait le score NaN, on l'écarte en le signalant
    empty = df_y_true.sum(axis=1) <= 0
    if empty.any():
        print(f"{int(empty.sum())} image(s) sans pixel évalué écartée(s) : {list(df_y_true.index[empty][:10])}")
        df_y_true, df_y_pred = df_y_true.loc[~empty], df_y_pred.loc[~empty]

    score = epsilon_kl_divergence(df_y_true, df_y_pred)
    print(f"Score (mean Kullback-Leibler divergence) = \n{score}")

    # ---------------- métriques supplémentaires ----------------
    class_names = list(df_y_true.columns)
    kl, cross_entropy, kl_contrib = per_image_scores(df_y_true, df_y_pred)
    per_class, presence_confusion = per_class_metrics(df_y_true, df_y_pred, kl_contrib,
                                                      class_names, args.presence_threshold)
    summary = summary_metrics(per_class, kl, cross_entropy, df_y_true, df_y_pred)
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 200):
        print(per_class[['mean_true', 'mean_pred', 'bias', 'kl_contribution',
                         'soft_f1', 'soft_iou', 'presence_f1']])
    print(pd.Series(summary).to_string())

    if args.out_csv is not None:
        print(f"Saving evaluation CSV to file {args.out_csv}")
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df = pd.Series({
            'gt_file':args.gt_file,
            'pred_file':args.pred_file,
            'score': score,
            **summary,   # métriques moyennes ajoutées à la suite des colonnes d'origine
        }).to_frame().T
        df.to_csv(args.out_csv, index=False)

        # fichiers détaillés, à côté du CSV de sortie
        out = Path(args.out_csv)
        stem = out.with_suffix('')
        per_class.to_csv(f'{stem}_per_class.csv', float_format='%.6f')
        pd.DataFrame({'kl': kl, 'cross_entropy': cross_entropy}, index=df_y_true.index)\
            .sort_values('kl', ascending=False)\
            .to_csv(f'{stem}_per_image.csv', float_format='%.6f')
        presence_confusion.to_csv(f'{stem}_presence_confusion.csv')
        plot_presence_confusion(presence_confusion, per_class, args.presence_threshold,
                                f'{stem}_presence_confusion.png')
        print(f"Détails : {stem}_per_class.csv, {stem}_per_image.csv, "
              f"{stem}_presence_confusion.csv / .png")
