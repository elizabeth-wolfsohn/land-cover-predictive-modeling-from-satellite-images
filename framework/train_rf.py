
"""Train the Random Forest."""

from pathlib import Path
import argparse
import datetime
import time

import numpy as np
import yaml
from tqdm import tqdm

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer


from framework.dataset import LandCoverData as LCD
from framework.models.random_forest import (
    LandCoverRF, read_image, read_mask,
    mask_path_from_image, extract_features
)



def sample_pixels(image_path, n_per_image, rng, rare_class_idx):
    """Sample pixels (features, labels) from one image."""

    img = read_image(image_path)
    mask = read_mask(mask_path_from_image(image_path))
    flat = mask.ravel()
    idx = np.flatnonzero(
        ~np.isin(flat, LCD.IGNORED_CLASSES_IDX)
    )
    if len(idx) == 0:
        return None, None
    selected_idx = []
    for class_id in np.unique(flat[idx]):
        # Indices des pixels de cette classe
        class_idx = idx[flat[idx] == class_id]
        # Keep all pixels for rare classes
        if class_id in rare_class_idx:
            sampled = class_idx
        # Other classes --> random sampling
        else:
            sampled = rng.choice(
                class_idx,
                size=min(n_per_image, len(class_idx)),
                replace=False
            )
        selected_idx.append(sampled)
    idx = np.concatenate(selected_idx)
    pixels = img.reshape(-1, img.shape[-1])[idx]
    feats = extract_features(
        pixels[:, None, :]
    ).reshape(len(idx), -1)
    return feats, flat[idx]



def stratified_image_selection(train_files, max_images, seed):
    """Select images using multilabel stratified sampling."""

    n_total = len(train_files)

    if max_images is None or max_images >= n_total:
        return train_files

    if max_images <= 0:
        raise ValueError("max_images must be positive.")

    # Build multilabel information for each image
    image_labels = []

    for image_path in tqdm(train_files, desc="Analyzing image classes for stratified image selection"):
        mask = read_mask(mask_path_from_image(image_path))

        present_classes = np.unique(mask)

        present_classes = [
            int(c) for c in present_classes
            if c not in LCD.IGNORED_CLASSES_IDX
        ]

        image_labels.append(present_classes)

    # Convert class lists into a binary matrix
    mlb = MultiLabelBinarizer(
        classes=np.arange(LCD.N_CLASSES)
    )

    Y = mlb.fit_transform(image_labels)

    # Stratified selection
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=max_images,
        random_state=seed
    )

    _, selected_idx = next(
        splitter.split(np.zeros(n_total), Y)
    )

    selected_files = [
        train_files[i] for i in sorted(selected_idx)
    ]

    print(f"Selected images: {len(selected_files)}/{n_total}")

    return selected_files


def parse_args():
    p = argparse.ArgumentParser('RF training')
    p.add_argument('--config', '-c', type=Path, required=True)
    return p.parse_args()


if __name__ == '__main__':

    # --- Load configuration ---
    args = parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # --- Configuration parameters ---
    REQUIRED_KEYS = [
        'dataset_folder', 'xp_rootdir', 'backend', 'seed',
        'n_per_image', 'max_images',
        'n_estimators',  'min_samples_leaf', 'max_features',
        'class_weight', 'n_jobs',        # CPU uniquement
        'n_bins',  "max_depth"     # GPU uniquement
    ]
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Clés manquantes dans {args.config} : {missing}")

    dataset_folder = Path(cfg['dataset_folder']).expanduser()
    xp_rootdir = Path(cfg['xp_rootdir']).expanduser()

    backend = cfg['backend']
    seed = cfg['seed']

    n_per_image = cfg['n_per_image']
    max_images = cfg['max_images']            # null = toutes les images

    n_estimators = cfg['n_estimators']
    min_samples_leaf = cfg['min_samples_leaf']
    max_features = cfg['max_features']
    max_depth = cfg['max_depth']

    class_weight = cfg['class_weight']        # CPU uniquement
    n_jobs = cfg['n_jobs']                    # CPU uniquement
    n_bins = cfg['n_bins']                    # GPU uniquement

    rare_class_idx=cfg['rare_class_idx'] 
    max_class_weight = cfg['max_class_weight']  


    print(f'Configuration: {cfg}')

    rng = np.random.default_rng(seed)

    # --- Load training images ---
    train_files = sorted(
        dataset_folder.glob('train/images/*.tif')
    )

    if not train_files:
        raise ValueError("Aucune image trouvée dans train/images/")

    if max_images is not None:
        train_files = stratified_image_selection(
            train_files,
            max_images,
            seed
    )

    print(f'Number of training images: {len(train_files)}')

    # --- Experiment folder ---
    xp_rootdir.mkdir(parents=True, exist_ok=True)

    xp_dir = xp_rootdir / datetime.datetime.now().strftime(
        "%d-%m-%Y_%H-%M-%S"
    )

    xp_dir.mkdir(parents=True)

    # --- Save experiment configuration ---
    config_out = xp_dir / 'config.yaml'

    with open(config_out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    print(f'Configuration saved to {config_out}')

    backend = cfg.get('backend', 'gpu')

    # --- Build the pixel training set ---
    t0 = time.perf_counter()

    Xs, ys = [], []

    for f in tqdm(train_files, desc='Sampling'):

        X, y = sample_pixels(f, n_per_image, rng, rare_class_idx)

        if X is not None:
            Xs.append(X)
            ys.append(y)

    if not Xs:
        raise ValueError("Aucun pixel valide n'a été échantillonné.")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    print(f'Class counts before sampling: {LCD.TRAIN_CLASS_COUNTS}')
    print(f'Sampling: {time.perf_counter() - t0:.0f}s, X={X.shape}')
    print(f'Class counts after sampling: {np.bincount(y, minlength=LCD.N_CLASSES)}')

    # --- Train Random Forest ---
    t0 = time.perf_counter()

    rf = LandCoverRF(
    backend=backend,
    n_estimators=n_estimators,
    min_samples_leaf=min_samples_leaf,
    class_weight=class_weight,
    max_features=max_features,
    max_depth=max_depth,
    n_jobs=n_jobs,
    random_state=seed,
    n_bins=n_bins,
    max_class_weight=max_class_weight,
)
    rf.fit(X, y)

    print(f'Fit: {time.perf_counter() - t0:.0f}s')

    # --- Save model ---
    rf.save(xp_dir / 'rf.joblib')

    print(f'Model saved to {xp_dir / "rf.joblib"}')
    print(f'Experiment directory: {xp_dir}')