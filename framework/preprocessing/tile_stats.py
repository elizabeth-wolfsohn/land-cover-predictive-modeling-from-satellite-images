"""
Per-tile class statistics of the training masks.

Each 256x256 mask is split into a regular grid of (256 / tile_size)^2 tiles, and the
number of pixels of every class is counted in each tile. With tile_size=128, an image
gives 4 tiles, indexed row-major:

    +-------+-------+
    |   0   |   1   |
    +-------+-------+
    |   2   |   3   |
    +-------+-------+

The result is a CSV with one row per (sample_id, tile_idx). It is computed once and
reused by every training (see framework/preprocessing/sampling.py). With
tile_size=256 the CSV simply contains the per-image class counts.

Usage:
    ipython framework/preprocessing/tile_stats.py -- \
        --dataset-folder challenge-ens/data/dataset_UNZIPPED \
        --tile-size 128 \
        --out challenge-ens/data/tile_stats_128.csv
"""
from pathlib import Path
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tifffile import TiffFile
from tqdm import tqdm

from framework.dataset import LandCoverData as LCD

# columns holding the pixel counts, one per class, named after the classes
CLASS_COLS = list(LCD.CLASSES)


def tile_offsets(tile_size, img_size=LCD.IMG_SIZE):
    """(row, col) offsets of the tiles of an image, in row-major tile_idx order."""
    if img_size % tile_size != 0:
        raise ValueError(f"tile_size={tile_size} must divide the image size {img_size}")
    n = img_size // tile_size
    return [((i // n) * tile_size, (i % n) * tile_size) for i in range(n * n)]


def _mask_tile_counts(mask_path, tile_size):
    """Class pixel counts for every tile of one mask file."""
    with TiffFile(mask_path) as tif:
        mask = tif.asarray()
    mask = np.squeeze(mask)
    sample_id = int(Path(mask_path).stem)
    rows = []
    for tile_idx, (r, c) in enumerate(tile_offsets(tile_size, mask.shape[0])):
        tile = mask[r:r + tile_size, c:c + tile_size]
        counts = np.bincount(tile.ravel(), minlength=LCD.N_CLASSES)
        if len(counts) > LCD.N_CLASSES:
            raise ValueError(f"{mask_path}: unexpected class value {len(counts) - 1}")
        rows.append((sample_id, tile_idx, tile_size, r, c, *counts))
    return rows


def compute_tile_stats(dataset_folder, tile_size=128, sample_ids=None, n_workers=None):
    """Compute the per-tile class pixel counts of the training masks.

    Args:
        dataset_folder (path-like): root of the dataset (containing train/masks/)
        tile_size (int): side of the square tiles, must divide 256
        sample_ids (iterable[int], optional): restrict to these samples (default: all)
        n_workers (int, optional): number of reading threads (default: number of CPUs)
    Returns:
        pandas.DataFrame: columns sample_id, tile_idx, tile_size, row, col, <one column per class>
    """
    masks_dir = Path(dataset_folder).expanduser() / 'train' / 'masks'
    mask_paths = sorted(masks_dir.glob('*.tif'))
    if sample_ids is not None:
        wanted = set(int(i) for i in sample_ids)
        mask_paths = [p for p in mask_paths if int(p.stem) in wanted]
    if not mask_paths:
        raise FileNotFoundError(f"no mask found in {masks_dir}")

    # threads (not processes): I/O bound, and safe to use after TensorFlow is imported
    n_workers = n_workers or os.cpu_count()
    rows = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for r in tqdm(ex.map(lambda p: _mask_tile_counts(p, tile_size), mask_paths),
                      total=len(mask_paths), desc=f'tile stats ({tile_size}px)'):
            rows.extend(r)

    columns = ['sample_id', 'tile_idx', 'tile_size', 'row', 'col'] + CLASS_COLS
    df = pd.DataFrame(rows, columns=columns)
    df[CLASS_COLS] = df[CLASS_COLS].astype(np.int64)
    return df.sort_values(['sample_id', 'tile_idx']).reset_index(drop=True)


def load_or_compute_tile_stats(csv_path, dataset_folder, tile_size=128, n_workers=None):
    """Read the tile stats CSV, computing and caching it first if it does not exist."""
    csv_path = Path(csv_path).expanduser()
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        sizes = df['tile_size'].unique()
        if len(sizes) != 1 or int(sizes[0]) != tile_size:
            raise ValueError(
                f"{csv_path} was computed with tile_size={sizes}, but tile_size={tile_size} "
                f"is requested: point `tile_stats_csv` to another file or delete it"
            )
        return df
    print(f"Tile stats not found at {csv_path}: computing them (done once)")
    df = compute_tile_stats(dataset_folder, tile_size=tile_size, n_workers=n_workers)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved tile stats to {csv_path}")
    return df


def image_class_counts(tile_stats):
    """Per-image class pixel counts (sum over the tiles), indexed by sample_id."""
    return tile_stats.groupby('sample_id')[CLASS_COLS].sum()


def _parse_args():
    parser = argparse.ArgumentParser('Compute per-tile class statistics of the training masks')
    parser.add_argument('--dataset-folder', '-d', type=str, required=True,
                        help="root of the dataset (containing train/masks/)")
    parser.add_argument('--tile-size', '-t', type=int, default=128, help="tile side, must divide 256")
    parser.add_argument('--out', '-o', type=str, required=True, help="output CSV file")
    parser.add_argument('--workers', '-w', type=int, default=None, help="number of reading threads")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    df = compute_tile_stats(args.dataset_folder, tile_size=args.tile_size, n_workers=args.workers)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} tiles ({df.sample_id.nunique()} images) to {out}")
