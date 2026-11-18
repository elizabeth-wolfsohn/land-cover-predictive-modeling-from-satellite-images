"""
Build the tf.data training and validation datasets with the optional preprocessing.

Per-sample processing order (training):
    read tif -> [crop tile] -> normalize -> geometric aug (image + mask)
             -> spectral indices (from the geometrically augmented bands)
             -> photometric aug (bands only) -> concat [bands, indices]
Validation: read tif -> normalize -> concat [bands, indices]  (never augmented, never tiled)

With tiling enabled, two streams are batched separately (a batch cannot mix sizes) and
interleaved randomly:
    intact images 256x256, batches of `batch_size`
    tiles         128x128, batches of `tile_batch_size` (default: batch_size * 4, same
                                                        number of pixels per batch)
a fraction `tile_batch_fraction` of the batches being tile batches. The U-Net is fully
convolutional, so it is built with a (None, None, C) input and accepts both sizes.

One "epoch" = as many pixels as one pass over the manifest.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from framework.dataset import LandCoverData as LCD
from framework.dataset import parse_image, normalize
from framework.preprocessing.config import cfg_get, section_enabled
from framework.preprocessing import spectral as spectral_mod
from framework.preprocessing.augmentation import Augmenter, random_flip_rot90
from framework.preprocessing.sampling import build_manifest, INTACT_TILE_IDX
from framework.preprocessing.tile_stats import load_or_compute_tile_stats

AUTOTUNE = tf.data.experimental.AUTOTUNE


def n_input_channels(config):
    """Number of input channels of the model for this config."""
    spectral = spectral_mod.from_config(config)
    return LCD.N_CHANNELS + (spectral.n_channels if spectral else 0)


def make_train_map_fn(augmenter, spectral):
    """(uint16 image, uint8 mask) -> (float image with optional indices, mask)."""
    def train_map(image, mask):
        image, mask = normalize(image, mask)
        if augmenter is not None:
            image, mask = augmenter.geometric(image, mask)
        else:
            image, mask = random_flip_rot90(image, mask)  # the benchmark augmentation
        indices = spectral.compute(image) if spectral is not None else None
        if augmenter is not None:
            image = augmenter.photometric(image)
        if indices is not None:
            image = tf.concat([image, indices], axis=-1)
        return image, mask
    return train_map


def make_eval_map_fn(spectral):
    """(uint16 image, uint8 mask) -> (float image with optional indices, mask)."""
    def eval_map(image, mask):
        image, mask = normalize(image, mask)
        if spectral is not None:
            image = spectral.append(image)
        return image, mask
    return eval_map


def make_tile_parse_fn(tile_size):
    """(path, tile_idx) -> (image tile, mask tile); tile_idx row-major (see tile_stats.py)."""
    n = LCD.IMG_SIZE // tile_size

    def parse_tile(path, tile_idx):
        image, mask = parse_image(path)
        tile_idx = tf.cast(tile_idx, tf.int32)
        row, col = (tile_idx // n) * tile_size, (tile_idx % n) * tile_size
        image = image[row:row + tile_size, col:col + tile_size, :]
        mask = mask[row:row + tile_size, col:col + tile_size, :]
        image.set_shape([tile_size, tile_size, LCD.N_CHANNELS])
        mask.set_shape([tile_size, tile_size, 1])
        return image, mask
    return parse_tile


def _image_path(dataset_folder, sample_id):
    return str(Path(dataset_folder) / 'train' / 'images' / f'{int(sample_id)}.tif')


def _intact_stream(paths, batch_size, map_fn, n_cpus):
    return tf.data.Dataset.from_tensor_slices(paths)\
        .shuffle(len(paths), reshuffle_each_iteration=True)\
        .repeat()\
        .map(parse_image, num_parallel_calls=n_cpus)\
        .map(map_fn, num_parallel_calls=n_cpus)\
        .batch(batch_size, drop_remainder=True)


def _tile_stream(paths, tile_idx, tile_size, batch_size, map_fn, n_cpus):
    # shuffle the (cheap) file references, not the decoded images
    return tf.data.Dataset.from_tensor_slices((paths, tile_idx))\
        .shuffle(len(paths), reshuffle_each_iteration=True)\
        .repeat()\
        .map(make_tile_parse_fn(tile_size), num_parallel_calls=n_cpus)\
        .map(map_fn, num_parallel_calls=n_cpus)\
        .batch(batch_size, drop_remainder=True)


def build_datasets(config, train_files, val_files, xp_dir, n_cpus=AUTOTUNE):
    """Build the train and validation datasets.

    Args:
        config (YamlNamespace): the training config
        train_files (list[Path]): training image files (validation excluded)
        val_files (list[Path]): validation image files
        xp_dir (Path): experiment folder, the manifest and reports are written in it
        n_cpus (int): parallelism of the map calls
    Returns:
        train_dataset (tf.data.Dataset): infinite, batched, yields (image, mask)
        val_dataset (tf.data.Dataset): infinite, batched, yields (image, mask)
        info (dict): 'steps_per_epoch', 'input_shape', 'class_counts' (effective class
            pixel counts of the manifest, or None), 'description'
    """
    xp_dir = Path(xp_dir)
    batch_size = int(config.batch_size)
    spectral = spectral_mod.from_config(config)
    augmenter = Augmenter.from_config(config)
    n_channels = LCD.N_CHANNELS + (spectral.n_channels if spectral else 0)
    print(f"Spectral indices: {spectral}")
    print(f"Augmentation: {augmenter if augmenter else 'benchmark (flip + rot90)'}")

    train_map = make_train_map_fn(augmenter, spectral)
    eval_map = make_eval_map_fn(spectral)

    val_dataset = tf.data.Dataset.from_tensor_slices([str(f) for f in val_files])\
        .map(parse_image, num_parallel_calls=n_cpus)\
        .map(eval_map, num_parallel_calls=n_cpus)\
        .repeat()\
        .batch(batch_size)\
        .prefetch(AUTOTUNE)

    info = {'class_counts': None}

    if not section_enabled(config, 'tiling'):
        paths = [str(f) for f in train_files]
        train_dataset = _intact_stream(paths, batch_size, train_map, n_cpus)
        info['steps_per_epoch'] = len(paths) // batch_size
        info['input_shape'] = (LCD.IMG_SIZE, LCD.IMG_SIZE, n_channels)
        info['description'] = f"{len(paths)} whole images"
        return train_dataset.prefetch(AUTOTUNE), val_dataset, info

    # ---------------- tiling + re-sampling ----------------
    tile_size = int(cfg_get(config, 'tiling.tile_size', 128))
    k = (LCD.IMG_SIZE // tile_size) ** 2
    tile_batch_size = int(cfg_get(config, 'tiling.tile_batch_size', batch_size * k))
    tile_batch_fraction = float(cfg_get(config, 'tiling.tile_batch_fraction', 0.5))
    intact_fraction = float(cfg_get(config, 'tiling.intact_fraction', 0.3))

    seed = cfg_get(config, 'tiling.resample_seed', None)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 32))  # new draw every training
    rng = np.random.default_rng(int(seed))

    stats_csv = cfg_get(config, 'tiling.tile_stats_csv',
                        Path(config.dataset_folder).parent / f'tile_stats_{tile_size}.csv')
    tile_stats = load_or_compute_tile_stats(stats_csv, config.dataset_folder, tile_size)

    train_ids = [int(Path(f).stem) for f in train_files]
    manifest, report, summary, effective_counts = build_manifest(
        train_ids, tile_stats, intact_fraction, rng,
        resampling=bool(cfg_get(config, 'tiling.resampling.enabled', True)),
        min_pixel_fraction=float(cfg_get(config, 'tiling.resampling.min_pixel_fraction', 0.01)),
        threshold=float(cfg_get(config, 'tiling.resampling.threshold', 0.1)),
        max_repeat=float(cfg_get(config, 'tiling.resampling.max_repeat', 4.0)),
        max_ignored_fraction=float(cfg_get(config, 'tiling.max_ignored_fraction', 1.0)),
    )

    # save everything needed to reproduce / analyse this draw
    out_dir = xp_dir / 'preprocessing'
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / 'train_manifest.csv', index=False)
    report.to_csv(out_dir / 'class_distribution_report.csv')
    with open(out_dir / 'resampling_summary.yaml', 'w') as f:
        yaml.safe_dump({'resample_seed': int(seed), 'tile_size': tile_size, **summary}, f, sort_keys=False)
    with pd.option_context('display.width', 200, 'display.float_format', '{:.4f}'.format):
        print(f"Resampling (seed {seed}): {summary}")
        print(report[['natural_share', 'effective_share', 'share_ratio',
                      'class_repeat_factor', 'pool_tiles_with_class']])

    intact = manifest[manifest['tile_idx'] == INTACT_TILE_IDX]
    tiles = manifest[manifest['tile_idx'] != INTACT_TILE_IDX]
    n_intact, n_tiles = len(intact), len(tiles)
    if n_intact < batch_size:
        tile_batch_fraction = 1.0
    if n_tiles < tile_batch_size:
        tile_batch_fraction = 0.0
    if n_intact < batch_size and n_tiles < tile_batch_size:
        raise ValueError("not enough images / tiles in the manifest for one batch")

    streams, weights = [], []
    if tile_batch_fraction > 0:
        streams.append(_tile_stream(
            [_image_path(config.dataset_folder, i) for i in tiles['sample_id']],
            tiles['tile_idx'].to_numpy(np.int32), tile_size, tile_batch_size, train_map, n_cpus))
        weights.append(tile_batch_fraction)
    if tile_batch_fraction < 1:
        streams.append(_intact_stream(
            [_image_path(config.dataset_folder, i) for i in intact['sample_id']],
            batch_size, train_map, n_cpus))
        weights.append(1.0 - tile_batch_fraction)
    train_dataset = streams[0] if len(streams) == 1 else \
        tf.data.Dataset.sample_from_datasets(streams, weights=weights)

    manifest_pixels = n_intact * LCD.IMG_SIZE ** 2 + n_tiles * tile_size ** 2
    pixels_per_step = tile_batch_fraction * tile_batch_size * tile_size ** 2 \
        + (1 - tile_batch_fraction) * batch_size * LCD.IMG_SIZE ** 2
    steps = cfg_get(config, 'tiling.steps_per_epoch', None)
    info['steps_per_epoch'] = int(steps) if steps else max(1, int(manifest_pixels // pixels_per_step))
    info['input_shape'] = (None, None, n_channels)
    info['class_counts'] = effective_counts
    info['description'] = (f"{n_intact} intact images ({LCD.IMG_SIZE}px, batch {batch_size}) + "
                           f"{n_tiles} tiles ({tile_size}px, batch {tile_batch_size}), "
                           f"{tile_batch_fraction:.0%} tile batches, seed {seed}")
    return train_dataset.prefetch(AUTOTUNE), val_dataset, info
