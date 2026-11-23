"""Predict class-proportion vectors and pixel-wise maps with Random Forest."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from tifffile import imwrite

from framework.dataset import LandCoverData as LCD
from framework.models.random_forest import LandCoverRF, read_image
import yaml


if __name__ == '__main__':

    p = argparse.ArgumentParser('RF inference')
    p.add_argument('--config', '-c', type=Path, required=True)
    cli = p.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    dataset_folder = Path(cfg['dataset_folder']).expanduser()
    xp_rootdir = Path(cfg['xp_rootdir']).expanduser()


    if cfg['xp_name'] == 'last':
        experiments = [
            d for d in xp_rootdir.iterdir()
            if d.is_dir() and (d / 'rf.joblib').is_file()
        ]

        if not experiments:
            raise FileNotFoundError(
                f"Aucun modèle rf.joblib trouvé dans {xp_rootdir}"
            )

        xp_dir = max(experiments, key=lambda d: d.stat().st_mtime)

        print(f"Dernière expérience sélectionnée : {xp_dir.name}")

    else:
        xp_dir = xp_rootdir / cfg['xp_name']

    assert xp_dir.is_dir(), xp_dir
    assert cfg['set'] in ('train', 'val', 'test')

    stride = cfg.get('stride', 1)
    set_name = cfg['set']

    if set_name == 'train':
        files = sorted(dataset_folder.glob('train/images/*.tif'))
    elif set_name == 'val':
        files = sorted(dataset_folder.glob('validation/images/*.tif'))
    else:
        files = sorted(dataset_folder.glob('test/images/*.tif'))

    if not files:
        raise ValueError(
            f"Aucune image trouvée pour le jeu {set_name} "
            f"dans {dataset_folder}"
        )

    rf = LandCoverRF.load(xp_dir / 'rf.joblib')

    print(f"Backend du modèle : {rf.backend}")

    # Dossier de sauvegarde des cartes de prédiction pixel par pixel
    pred_mask_dir = xp_dir / f'rf_{set_name}_pred_masks'
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    y_pred = []

    for image_path in tqdm(files, desc="Predicting"):

        img = read_image(image_path)

        # 1. Prédiction pixel par pixel
        predicted_mask = rf.predict_image(img, stride=stride)

        # 2. Sauvegarde de la carte prédite
        out_mask = pred_mask_dir / image_path.name
        imwrite(out_mask, predicted_mask.astype(np.uint8))

        # 3. Proportions calculées à partir du masque déjà prédit
        counts = np.bincount(predicted_mask.ravel(), minlength=LCD.N_CLASSES)
        y_pred.append(counts / counts.sum())

    # Sauvegarde des proportions par image
    y_pred = np.stack(y_pred)

    ids = pd.Series(
        [int(f.stem) for f in files],
        name='sample_id',
        dtype='uint32'
    )

    df = pd.DataFrame(y_pred, index=ids, columns=LCD.CLASSES)

    out_csv = xp_dir / f'rf_{set_name}_predicted.csv'

    print(f"Saving prediction CSV to {out_csv}")
    df.to_csv(out_csv, index=True, index_label='sample_id')

    print(f"Pixel-wise predictions saved in {pred_mask_dir}")
