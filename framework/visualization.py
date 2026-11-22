"""
Visualization helpers: dataset samples, predicted land cover maps and training curves.
All figures are saved as PNG files (non-interactive backend, safe to use during training).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from tifffile import TiffFile

from framework.dataset import LandCoverData as LCD

# French display names of the classes
CLASS_NAMES_FR = {
    0: 'Pas de donnée',
    1: 'Nuages',
    2: 'Artificiel',
    3: 'Cultures',
    4: 'Feuillus',
    5: 'Conifères',
    6: 'Herbacé',
    7: 'Sol nu / roche',
    8: 'Neige',
    9: 'Eau',
}
EVAL_CLASSES = [c for c in range(LCD.N_CLASSES) if c not in LCD.IGNORED_CLASSES_IDX]

_COLORS = np.stack([LCD.CLASSES_COLORPALETTE[c] for c in range(LCD.N_CLASSES)]) / 255.
MASK_CMAP = ListedColormap(_COLORS)
MASK_NORM = BoundaryNorm(np.arange(-0.5, LCD.N_CLASSES), LCD.N_CLASSES)


def _stretch(image, bands):
    """Contrast stretch (2-98 percentiles) of the selected bands for display."""
    img = np.asarray(image, dtype=np.float32)[..., bands]
    lo, hi = np.percentile(img, (2, 98), axis=(0, 1))
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-8), 0, 1)


def rgb(image):
    """True colours (R, G, B)."""
    return _stretch(image, [0, 1, 2])


def false_color(image):
    """False colours (NIR, R, G): vegetation appears red, water dark, built-up areas cyan/grey."""
    return _stretch(image, [3, 0, 1])


def show_mask(ax, mask):
    ax.imshow(np.squeeze(mask), cmap=MASK_CMAP, norm=MASK_NORM, interpolation='nearest')


def class_legend(fig, classes=EVAL_CLASSES, ncol=None, **kwargs):
    handles = [Patch(facecolor=_COLORS[c], edgecolor='k', label=CLASS_NAMES_FR[c]) for c in classes]
    fig.legend(handles=handles, loc='lower center', ncol=ncol or len(classes), frameon=False, **kwargs)


def class_distribution(mask):
    """Proportions of the evaluated classes in a mask (same as the challenge targets)."""
    counts = np.bincount(np.asarray(mask).ravel().astype(np.int64), minlength=LCD.N_CLASSES)
    counts = counts[EVAL_CLASSES].astype(np.float64)
    return counts / max(counts.sum(), 1.)


def kl_divergence(y_true, y_pred, eps=1e-7):
    y_true = np.asarray(y_true, np.float64) / np.sum(y_true)
    y_pred = np.asarray(y_pred, np.float64) / np.sum(y_pred)
    return float(np.sum((y_true + eps) * np.log((y_true + eps) / (y_pred + eps))))


def open_file(path):
    """Open a file with the default application (Windows only, ignored elsewhere)."""
    try:
        if sys.platform.startswith('win'):
            os.startfile(str(path))
    except OSError as e:
        print(f"Could not open {path}: {e}")


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def load_sample(image_path):
    """Load an image (uint16, H x W x 4) and its mask (H x W) from the train folder."""
    image_path = Path(image_path)
    mask_path = image_path.parent.parent/'masks'/image_path.name
    with TiffFile(image_path) as tif:
        image = tif.asarray()
    mask = None
    if mask_path.is_file():
        with TiffFile(mask_path) as tif:
            mask = tif.asarray()
    return image, mask


def plot_dataset_samples(image_files, save_path, num=10):
    """Grid with, for every sample: true colours, false colours (NIR) and the land cover mask."""
    image_files = list(image_files)[:num]
    n = len(image_files)
    fig, axs = plt.subplots(3, n, figsize=(2.2 * n, 7.4), squeeze=False)
    for j, f in enumerate(image_files):
        image, mask = load_sample(f)
        axs[0, j].imshow(rgb(image))
        axs[0, j].set_title(f'#{Path(f).stem}', fontsize=9)
        axs[1, j].imshow(false_color(image))
        if mask is not None:
            show_mask(axs[2, j], mask)
        for i in range(3):
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
    axs[0, 0].set_ylabel('Couleurs (RGB)')
    axs[1, 0].set_ylabel('Fausses couleurs\n(NIR-R-G)')
    axs[2, 0].set_ylabel('Masque vérité')
    fig.suptitle(f'{n} échantillons du jeu d\'entraînement (pixels de 10 m, images de 2,56 km)')
    class_legend(fig, ncol=8, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    return save_path


def plot_predictions(images, masks, pred_probs, save_path, title=None, sample_ids=None):
    """For every image: RGB, ground truth, prediction, errors map and class proportions.
    Args:
        images: (N, H, W, 4) images (any scale)
        masks: (N, H, W[, 1]) ground-truth masks, or None (test set)
        pred_probs: (N, H, W, C) predicted probabilities
    """
    images = np.asarray(images)
    pred_probs = np.asarray(pred_probs)
    pred_masks = pred_probs.argmax(-1)
    n = len(images)
    has_gt = masks is not None
    ncols = 5 if has_gt else 3
    fig, axs = plt.subplots(n, ncols, figsize=(3.3 * ncols, 3.2 * n), squeeze=False,
                            gridspec_kw={'width_ratios': [1] * (ncols - 1) + [1.6]})
    x = np.arange(len(EVAL_CLASSES))
    for i in range(n):
        col = 0
        axs[i, col].imshow(rgb(images[i]))
        axs[i, col].set_title('Image' + (f' #{sample_ids[i]}' if sample_ids is not None else ''))
        col += 1
        if has_gt:
            mask = np.squeeze(np.asarray(masks[i]))
            show_mask(axs[i, col], mask); axs[i, col].set_title('Vérité terrain'); col += 1
        show_mask(axs[i, col], pred_masks[i]); axs[i, col].set_title('Prédiction du modèle'); col += 1
        if has_gt:
            valid = ~np.isin(mask, LCD.IGNORED_CLASSES_IDX)
            wrong = (pred_masks[i] != mask) & valid
            err = np.ones(mask.shape + (3,))
            err[wrong] = (0.85, 0.1, 0.1)
            err[~valid] = (0.6, 0.6, 0.6)
            acc = 1. - wrong.sum() / max(valid.sum(), 1)
            axs[i, col].imshow(err)
            axs[i, col].set_title(f'Erreurs (rouge) - précision {acc:.1%}')
            col += 1
        # class proportions (soft aggregation of the probabilities)
        p_pred = pred_probs[i].mean((0, 1))[EVAL_CLASSES]
        p_pred = p_pred / p_pred.sum()
        ax = axs[i, col]
        if has_gt:
            p_true = class_distribution(mask)
            ax.bar(x - 0.2, p_true, 0.4, color=_COLORS[EVAL_CLASSES], edgecolor='k', label='vérité')
            ax.bar(x + 0.2, p_pred, 0.4, color=_COLORS[EVAL_CLASSES], edgecolor='k', hatch='//',
                   alpha=0.6, label='prédit')
            ax.set_title(f'Proportions - KL = {kl_divergence(p_true, p_pred):.4f}')
            ax.legend(fontsize=8)
        else:
            ax.bar(x, p_pred, 0.6, color=_COLORS[EVAL_CLASSES], edgecolor='k')
            ax.set_title('Proportions prédites')
        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_NAMES_FR[c] for c in EVAL_CLASSES], rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1)
        for a in axs[i, :-1]:
            a.set_xticks([]); a.set_yticks([])
    if title:
        fig.suptitle(title)
    class_legend(fig, ncol=8, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97 if title else 1))
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


def plot_training_curves(history, save_path):
    """Loss, pixel accuracy and KL divergence (challenge metric) for train and validation.
    Args:
        history (pandas.DataFrame or dict of lists): logs per epoch (as written by CSVLogger)
    """
    df = pd.DataFrame(history)
    epochs = df['epoch'] + 1 if 'epoch' in df else np.arange(1, len(df) + 1)
    panels = [('loss', 'Loss (entropie croisée)'),
              ('accuracy', 'Précision pixel (accuracy)'),
              ('kl', 'Divergence KL (métrique du challenge)')]
    panels = [(k, t) for k, t in panels if k in df]
    fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2), squeeze=False)
    for ax, (key, title) in zip(axs[0], panels):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(epochs, df[key], 'o-', ms=3, label='entraînement')
        if f'val_{key}' in df:
            ax.plot(epochs, df[f'val_{key}'], 'o-', ms=3, label='validation')
            best = df[f'val_{key}'].idxmax() if key == 'accuracy' else df[f'val_{key}'].idxmin()
            ax.axvline(epochs[best], color='grey', ls='--', lw=0.8)
            title += f"\nmeilleure val : {df[f'val_{key}'][best]:.4f} (époque {epochs[best]})"
        if key == 'kl':
            ax.axhline(0.0593, color='red', ls=':', lw=1, label='benchmark U-Net (test public)')
        ax.set_title(title); ax.set_xlabel('époque'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    return save_path
