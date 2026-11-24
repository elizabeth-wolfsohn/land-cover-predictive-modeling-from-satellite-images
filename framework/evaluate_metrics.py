"""
Évaluation détaillée d'une expérience sur la validation et le test.

Pour chaque ensemble (val, test), le script fait prédire le modèle sur les images,
puis calcule, en excluant les 2 premières classes (no_data, clouds) :
  - par classe : précision, rappel, F1, IoU (au pixel)               -> per_class_metrics_<split>.csv
  - global : moyennes (macro et pondérées), exactitude, KL, entropie croisée
                                                                       -> summary_metrics.csv
  - la matrice de confusion (comptes et normalisée par ligne)         -> confusion_matrix_<split>*.csv / .png
  - la KL de chaque image, pour repérer les pires                     -> per_image_kl_<split>.csv

Les vraies proportions utilisées pour la KL sont lues dans les CSV de labels
(format y_train : sample_id puis une colonne par classe). Les métriques au pixel
utilisent les masques du dossier train/masks.

Si `spectral_indices.enabled` vaut true dans le train_config.yaml de l'expérience,
les indices listés (NDVI, NDWI) sont ajoutés à chaque image comme canaux
supplémentaires, comme à l'entraînement.

Usage, depuis la racine du projet :
    ipython framework/evaluate_metrics.py
"""
import sys
from pathlib import Path

# ============================== À RENSEIGNER ==============================
EXPERIMENT_PATH = Path('challenge-ens/code/experiments/21-11-2026_16:15:13')
VAL_LABELS_CSV = Path('local_dataset/csv_files/validation_labels.csv')
TEST_LABELS_CSV = Path('local_dataset/csv_files/test_labels.csv')
# ==========================================================================

# Options (les valeurs par défaut conviennent en général)
DATASET_FOLDER = "challenge-ens/data/dataset_UNZIPPED"   # None = lu dans EXPERIMENT_PATH/train_config.yaml ; sinon ex. Path('challenge-ens/data/DATASET_UNZIPPED')
CHECKPOINT_EPOCH = 90 # None = l'époque avec la meilleure val_kl dans fit_logs.csv
BATCH_SIZE = 32
N_EXCLUDED = 2          # les N premières classes sont exclues (no_data, clouds)
KL_EPS = 1e-8           # epsilon de la métrique officielle du challenge

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')   # dessin sans fenêtre : pas d'erreur tkinter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf

# permet de lancer le script avec python comme avec ipython depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from framework.dataset import parse_image, load_image_test  # noqa: E402


# --------------------------------------------------------------------------
# Expérience : dossier des données et checkpoint
# --------------------------------------------------------------------------
def load_train_config(xp_dir: Path):
    """Contenu de train_config.yaml de l'expérience (dict), ou None s'il n'existe pas."""
    cfg_path = Path('train_config.yaml')
    if not cfg_path.is_file():
        print(f"{cfg_path} introuvable : pas d'indices spectraux, DATASET_FOLDER doit être renseigné")
        return None
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def resolve_dataset_folder(train_cfg) -> Path:
    """Dossier contenant train/images et train/masks."""
    folder = DATASET_FOLDER
    if folder is None:
        if not train_cfg or 'dataset_folder' not in train_cfg:
            raise FileNotFoundError("dataset_folder absent de train_config.yaml : "
                                    "renseigne DATASET_FOLDER en haut du script")
        folder = train_cfg['dataset_folder']
    folder = Path(folder).expanduser()
    if not (folder / 'train' / 'images').is_dir():
        raise FileNotFoundError(f"{folder}/train/images introuvable (dossier courant : {Path.cwd()})")
    return folder


def select_checkpoint(xp_dir: Path, epoch=None):
    """Renvoie (chemin, époque) : l'époque demandée, ou celle de la meilleure val_kl."""
    if epoch is None:
        logs = pd.read_csv(xp_dir / 'fit_logs.csv')
        column = 'val_kl' if 'val_kl' in logs.columns else 'val_loss'
        best = logs.loc[logs[column].idxmin()]
        epoch = int(best['epoch']) + 1      # la colonne epoch du CSV commence à 0
        print(f"Checkpoint choisi : époque {epoch} (meilleure {column} = {best[column]:.5f})")
    for candidate in (xp_dir / 'checkpoints' / f'epoch{epoch}',
                      xp_dir / 'checkpoints' / f'epoch{epoch}.keras'):
        if candidate.exists():
            return candidate, int(epoch)
    raise FileNotFoundError(f"Pas de checkpoint pour l'époque {epoch} dans {xp_dir / 'checkpoints'}")


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------
def read_labels(csv_path: Path) -> pd.DataFrame:
    """CSV au format y_train : sample_id en première colonne, puis une colonne par classe."""
    labels = pd.read_csv(csv_path, index_col=0)
    labels.index = labels.index.astype(int)
    if labels.shape[1] <= N_EXCLUDED:
        raise ValueError(f"{csv_path} : {labels.shape[1]} colonnes de classes, il en faut plus de {N_EXCLUDED}")
    return labels


def make_dataset(ids, dataset_folder: Path, eval_map) -> tf.data.Dataset:
    """Images et masques dans l'ordre de `ids`, préparés comme la validation à l'entraînement."""
    paths = [dataset_folder / 'train' / 'images' / f'{i}.tif' for i in ids]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) introuvable(s), ex. {missing[:3]}")
    return tf.data.Dataset.from_tensor_slices([str(p) for p in paths])\
        .map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(eval_map, num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(BATCH_SIZE)\
        .prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------
# Indices spectraux (NDVI, NDWI)
# --------------------------------------------------------------------------
# indice -> (bande a, bande b) : (a - b) / (a + b)
SPECTRAL_FORMULAS = {
    'ndvi': ('nir', 'red'),     # végétation : (NIR - R) / (NIR + R)
    'ndwi': ('green', 'nir'),   # eau        : (G - NIR) / (G + NIR)
}
DEFAULT_BAND_ORDER = {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}


def spectral_settings(train_cfg):
    """(indices, ordre des bandes) si spectral_indices.enabled dans train_config.yaml, sinon None."""
    section = (train_cfg or {}).get('spectral_indices') or {}
    if not section.get('enabled', False):
        return None
    indices = [str(name).lower() for name in (section.get('indices') or ['ndvi', 'ndwi'])]
    unknown = [name for name in indices if name not in SPECTRAL_FORMULAS]
    if unknown:
        raise ValueError(f"Indice(s) spectral(aux) inconnu(s) : {unknown} "
                         f"(disponibles : {list(SPECTRAL_FORMULAS)})")
    band_order = {**DEFAULT_BAND_ORDER, **(section.get('band_order') or {})}
    return indices, band_order


def _local_spectral_eval_map(indices, band_order):
    """Calcul de secours : indices sur les bandes brutes (avant normalisation), ajoutés après les 4 bandes."""
    def eval_map(image, mask):
        raw = tf.cast(image, tf.float32)
        extra = []
        for name in indices:
            band_a, band_b = SPECTRAL_FORMULAS[name]
            a, b = raw[..., band_order[band_a]], raw[..., band_order[band_b]]
            extra.append((a - b) / (a + b + 1e-6))
        image, mask = load_image_test(image, mask)
        return tf.concat([tf.cast(image, tf.float32), tf.stack(extra, axis=-1)], axis=-1), mask
    return eval_map


def make_eval_map(train_cfg):
    """Préparation des images : renvoie (fonction, description des indices ajoutés).

    Avec les indices activés, les fonctions du projet (framework/preprocessing, utilisées
    par build_datasets à l'entraînement) sont prises en priorité, pour une préparation
    identique. Sans elles, un calcul local de secours est utilisé et signalé.
    """
    settings = spectral_settings(train_cfg)
    if settings is None:
        print("Indices spectraux : désactivés (4 bandes)")
        return load_image_test, 'aucun'
    indices, band_order = settings
    description = ','.join(indices)
    try:
        from framework.utils import YamlNamespace
        from framework.preprocessing import spectral as spectral_mod
        from framework.preprocessing.pipeline import make_eval_map_fn
    except ImportError as error:
        print(f"ATTENTION : module du projet introuvable ({error}).\n"
              f"  Indices {description} calculés localement sur les bandes brutes, avant "
              f"normalisation (ordre des bandes : {band_order}).\n"
              "  Vérifie que c'est le même calcul qu'à l'entraînement, sinon les métriques seront faussées.")
        return _local_spectral_eval_map(indices, band_order), f'{description} (calcul local)'
    spectral = spectral_mod.from_config(YamlNamespace(train_cfg))
    print(f"Indices spectraux : {description} (fonctions du projet, identiques à l'entraînement)")
    return make_eval_map_fn(spectral), description


# --------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------
def kl_per_image(true_props, pred_props, eps=KL_EPS):
    """KL officielle par image ; NaN pour une image sans pixel évalué."""
    t_sum = true_props.sum(1, keepdims=True)
    t = np.divide(true_props, t_sum, out=np.zeros_like(true_props), where=t_sum > 0)
    p = pred_props / np.clip(pred_props.sum(1, keepdims=True), 1e-12, None)
    kl = np.sum((t + eps) * np.log((t + eps) / (p + eps)), axis=1)
    kl[t_sum[:, 0] == 0] = np.nan
    return kl


def per_class_metrics(cm: np.ndarray, names) -> pd.DataFrame:
    """Précision, rappel, F1 et IoU par classe à partir de la matrice (lignes = vraies classes)."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    def ratio(num, den):
        # même convention que scikit-learn (zero_division=0) : un ratio indéfini vaut 0
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    iou = ratio(tp, tp + fp + fn)
    # une classe ni présente ni prédite n'a pas de sens : NaN, exclue des moyennes
    absent = (tp + fp + fn) == 0
    for values in (precision, recall, f1, iou):
        values[absent] = np.nan
    total = cm.sum()
    return pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'support_pixels': cm.sum(axis=1),
        'true_share': cm.sum(axis=1) / total,
        'predicted_share': cm.sum(axis=0) / total,
    }, index=pd.Index(names, name='classe'))


def summarize(per_class: pd.DataFrame, cm: np.ndarray, kl_soft, kl_hard, ce_sum, n_pixels, epoch) -> dict:
    support = per_class['support_pixels'].to_numpy(float)
    weights = support / support.sum()

    def weighted(column):
        values = per_class[column].to_numpy(float)
        ok = ~np.isnan(values)
        return float(np.sum(values[ok] * weights[ok]) / weights[ok].sum())

    metrics = {
        'checkpoint_epoch': epoch,
        'n_images': int(np.sum(~np.isnan(kl_soft))),
        'n_pixels_evalues': int(n_pixels),
        'kl_softmax': float(np.nanmean(kl_soft)),   # proportions = moyenne des probabilités
        'kl_argmax': float(np.nanmean(kl_hard)),    # proportions = comptage des pixels après argmax
        'cross_entropy': float(ce_sum / n_pixels),
        'pixel_accuracy': float(np.trace(cm) / cm.sum()),
        'macro_precision': float(per_class['precision'].mean()),
        'macro_recall': float(per_class['recall'].mean()),
        'macro_f1': float(per_class['f1'].mean()),
        'mean_iou': float(per_class['iou'].mean()),
        'weighted_precision': weighted('precision'),
        'weighted_recall': weighted('recall'),
        'weighted_f1': weighted('f1'),
        'weighted_iou': weighted('iou'),
    }
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}


# --------------------------------------------------------------------------
# Évaluation d'un ensemble
# --------------------------------------------------------------------------
def evaluate_split(model, labels: pd.DataFrame, dataset_folder: Path, eval_map):
    ids = labels.index.to_numpy()
    n_classes = labels.shape[1]
    n_valid = n_classes - N_EXCLUDED
    cm = np.zeros((n_valid, n_valid), dtype=np.int64)
    ce_sum, n_pixels = 0.0, 0
    soft_props, hard_props = [], []

    for images, masks in make_dataset(ids, dataset_folder, eval_map):
        if images.shape[-1] != model.input_shape[-1]:
            raise ValueError(f"Le modèle attend {model.input_shape[-1]} canaux, les images préparées en ont "
                             f"{images.shape[-1]} : vérifie spectral_indices dans le train_config.yaml "
                             "de l'expérience (activé ou non, liste des indices)")
        probs = np.asarray(model.predict_on_batch(images), dtype=np.float64)    # (B, H, W, C)
        if probs.shape[-1] != n_classes:
            raise ValueError(f"Le modèle prédit {probs.shape[-1]} classes, le CSV en contient {n_classes}")
        masks = np.asarray(masks).reshape(probs.shape[:-1]).astype(np.int64)
        if masks.max() >= n_classes:
            raise ValueError(f"Classe {masks.max()} dans un masque, au-delà des {n_classes} classes du CSV")

        p_valid = probs[..., N_EXCLUDED:]                                      # (B, H, W, n_valid)
        pred = p_valid.argmax(axis=-1)                                         # argmax parmi les classes évaluées

        # proportions prédites par image, des deux façons
        soft = p_valid.mean(axis=(1, 2))
        soft_props.append(soft / soft.sum(axis=1, keepdims=True))
        hard = np.stack([np.bincount(p.ravel(), minlength=n_valid) for p in pred]).astype(float)
        hard_props.append(hard / hard.sum(axis=1, keepdims=True))

        # pixels évalués : vraie classe hors no_data / clouds
        valid = masks >= N_EXCLUDED
        true_idx = masks[valid] - N_EXCLUDED
        cm += np.bincount(true_idx * n_valid + pred[valid], minlength=n_valid ** 2)\
            .reshape(n_valid, n_valid)

        # entropie croisée, probabilités renormalisées sur les classes évaluées
        p_norm = p_valid / p_valid.sum(axis=-1, keepdims=True)
        target = np.clip(masks, N_EXCLUDED, None) - N_EXCLUDED
        p_true = np.take_along_axis(p_norm, target[..., None], axis=-1)[..., 0]
        ce_sum += float(-np.log(np.clip(p_true[valid], 1e-7, 1.0)).sum())
        n_pixels += int(valid.sum())

    soft_props = np.concatenate(soft_props)
    hard_props = np.concatenate(hard_props)
    true_props = labels.iloc[:, N_EXCLUDED:].to_numpy(dtype=np.float64)
    kl_soft = kl_per_image(true_props, soft_props)
    kl_hard = kl_per_image(true_props, hard_props)
    per_image = pd.DataFrame({'kl_softmax': kl_soft, 'kl_argmax': kl_hard},
                             index=pd.Index(ids, name='sample_id'))
    return cm, kl_soft, kl_hard, ce_sum, n_pixels, per_image


# --------------------------------------------------------------------------
# Matrice de confusion en image
# --------------------------------------------------------------------------
# échelle séquentielle bleue (clair = peu, foncé = beaucoup)
SEQUENTIAL_BLUE = ['#f3f7fc', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
                   '#256abf', '#184f95', '#0d366b']
INK_DARK, INK_LIGHT, INK_MUTED = '#1f1f1f', '#ffffff', '#6b6b6b'


def plot_confusion(cm: np.ndarray, names, title: str, path: Path):
    """Matrice normalisée par ligne : chaque ligne montre où vont les pixels d'une vraie classe."""
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = cm / cm.sum(axis=1, keepdims=True)
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.5, 0.85 * n + 1.8))
    cmap = LinearSegmentedColormap.from_list('sequential_blue', SEQUENTIAL_BLUE)
    image = ax.imshow(norm * 100, cmap=cmap, vmin=0, vmax=100)

    for i in range(n):
        for j in range(n):
            value = norm[i, j]
            if np.isnan(value):
                ax.text(j, i, '–', ha='center', va='center', color=INK_MUTED, fontsize=9)
                continue
            if cm[i, j] == 0:   # aucun pixel : « 0 » discret, distinct d'une valeur arrondie à 0.0
                ax.text(j, i, '0', ha='center', va='center', color=INK_MUTED, fontsize=8)
                continue
            color = INK_LIGHT if value >= 0.5 else INK_DARK
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{value * 100:.1f}', ha='center', va='center',
                    color=color, fontsize=9, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha='right', color=INK_DARK)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, color=INK_DARK)
    ax.set_xlabel('Classe prédite', color=INK_DARK)
    ax.set_ylabel('Classe réelle', color=INK_DARK)
    ax.set_title(title, loc='left', color=INK_DARK, fontsize=11)
    # 2 px de séparation blanche entre les cases, sans cadre
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label('% des pixels de la classe réelle (diagonale = rappel)', color=INK_DARK)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, colors=INK_MUTED)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    xp_dir = Path(EXPERIMENT_PATH).expanduser()
    if not xp_dir.is_dir():
        raise FileNotFoundError(f"Expérience introuvable : {xp_dir.resolve()}")
    train_cfg = load_train_config(xp_dir)
    dataset_folder = resolve_dataset_folder(train_cfg)
    eval_map, spectral_description = make_eval_map(train_cfg)
    ckpt_path, epoch = select_checkpoint(xp_dir, CHECKPOINT_EPOCH)
    # compile=False : seuls l'architecture et les poids sont nécessaires
    model = tf.keras.models.load_model(str(ckpt_path), compile=False)

    out_dir = xp_dir / f'evaluation_epoch{epoch}'
    out_dir.mkdir(exist_ok=True)
    summaries = {}

    for split, csv_path in (('val', VAL_LABELS_CSV), ('test', TEST_LABELS_CSV)):
        labels = read_labels(Path(csv_path).expanduser())
        names = list(labels.columns[N_EXCLUDED:])
        print(f"\n=== {split} : {len(labels)} images ===")
        cm, kl_soft, kl_hard, ce_sum, n_pixels, per_image = evaluate_split(
            model, labels, dataset_folder, eval_map)

        per_class = per_class_metrics(cm, names)
        per_class.to_csv(out_dir / f'per_class_metrics_{split}.csv', float_format='%.6f')
        summaries[split] = summarize(per_class, cm, kl_soft, kl_hard, ce_sum, n_pixels, epoch)
        summaries[split]['spectral_indices'] = spectral_description

        cm_df = pd.DataFrame(cm, index=pd.Index(names, name='classe_reelle'),
                             columns=pd.Index(names, name='classe_predite'))
        cm_df.to_csv(out_dir / f'confusion_matrix_{split}.csv')
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_df.div(cm_df.sum(axis=1), axis=0)\
                .to_csv(out_dir / f'confusion_matrix_{split}_normalized.csv', float_format='%.6f')
        n_pixels_txt = f"{n_pixels:,}".replace(',', ' ')
        plot_confusion(cm, names, f"Matrice de confusion – {split} (époque {epoch}, {n_pixels_txt} pixels)",
                       out_dir / f'confusion_matrix_{split}.png')
        per_image.sort_values('kl_softmax', ascending=False)\
            .to_csv(out_dir / f'per_image_kl_{split}.csv', float_format='%.6f')

        with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 160):
            print(per_class[['precision', 'recall', 'f1', 'iou', 'true_share', 'predicted_share']])

    # séries de type object : les entiers (époque, nombre d'images) restent des entiers
    summary = pd.DataFrame({split: pd.Series(values, dtype=object)
                            for split, values in summaries.items()})
    summary.index.name = 'metrique'
    summary.to_csv(out_dir / 'summary_metrics.csv')
    print('\n', summary)
    print(f"\nRésultats écrits dans {out_dir}")


if __name__ == '__main__':
    main()
