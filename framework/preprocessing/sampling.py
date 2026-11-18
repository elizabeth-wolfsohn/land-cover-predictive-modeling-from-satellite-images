"""
Build the training manifest of one training run: which images are kept whole (256x256)
and which tiles (e.g. 128x128) are used, with how many repetitions.

At every training, with a new `resample_seed`:
  1. the training images (validation excluded, never touched) are shuffled;
  2. a fraction `intact_fraction` of them is kept whole -> 'intact' entries;
  3. the other images form the 'pool': they are split into tiles, and the tiles are
     re-sampled with Repeat Factor Sampling (Gupta et al., LVIS, 2019) so that tiles
     containing rare classes appear several times:

        f(c)  = fraction of pool tiles where class c covers >= min_pixel_fraction of the tile
        r(c)  = max(1, sqrt(threshold / f(c)))            (class repeat factor)
        r(t)  = min(max_repeat, max_{c present in t} r(c)) (tile repeat factor)
        copies(t) = floor(r(t)) + Bernoulli(r(t) - floor(r(t)))

     The stochastic rounding makes every training see a different set of tiles.

Splitting is done per *image*: a tile never comes from a validation image.

Splitting an image into tiles does not change the class proportions by itself: the
re-balancing comes from step 3, which is possible because tiles are "purer" than
whole images (a rare class covers a larger share of a 128px tile).
"""
import numpy as np
import pandas as pd

from framework.dataset import LandCoverData as LCD
from framework.preprocessing.tile_stats import CLASS_COLS, image_class_counts

IGNORED_IDX = list(LCD.IGNORED_CLASSES_IDX)
VALID_IDX = [i for i in range(LCD.N_CLASSES) if i not in IGNORED_IDX]
INTACT_TILE_IDX = -1


def class_presence(tile_counts, min_pixel_fraction):
    """Boolean (n_tiles, n_classes): class covers >= min_pixel_fraction of the tile.
    Ignored classes are never considered present."""
    pixels = tile_counts.sum(1, keepdims=True).clip(min=1)
    presence = (tile_counts / pixels) >= min_pixel_fraction
    presence[:, IGNORED_IDX] = False
    # a class with 0 pixel is never present, even if min_pixel_fraction == 0
    presence &= tile_counts > 0
    return presence


def class_repeat_factors(presence, threshold):
    """Per-class repeat factor r(c) = max(1, sqrt(threshold / f(c))) and frequency f(c)."""
    freq = presence.mean(0) if len(presence) else np.zeros(presence.shape[1])
    safe = np.where(freq > 0, freq, 1.0)
    r = np.where(freq > 0, np.maximum(1.0, np.sqrt(threshold / safe)), 1.0)
    r[IGNORED_IDX] = 1.0
    return r, freq


def tile_repeat_factors(presence, class_r, max_repeat):
    """Per-tile repeat factor: max over present classes of r(c), clipped to max_repeat."""
    rf = np.where(presence, class_r[None, :], 1.0).max(1)
    return np.minimum(rf, max_repeat)


def stochastic_round(x, rng):
    """floor(x) + Bernoulli(frac(x)): an integer with expectation x."""
    floor = np.floor(x)
    return (floor + (rng.random(len(x)) < (x - floor))).astype(np.int64)


def build_manifest(train_ids, tile_stats, intact_fraction, rng,
                   resampling=True, min_pixel_fraction=0.01, threshold=0.1,
                   max_repeat=4.0, max_ignored_fraction=1.0):
    """Draw the manifest of one training run.

    Args:
        train_ids (iterable[int]): ids of the training images (validation excluded)
        tile_stats (pandas.DataFrame): output of tile_stats.compute_tile_stats
        intact_fraction (float): fraction of train images kept whole, in [0, 1]
        rng (numpy.random.Generator): random generator (seeded by `resample_seed`)
        resampling (bool): apply Repeat Factor Sampling on the pool tiles
        min_pixel_fraction (float): share of a tile a class must cover to count as present
        threshold (float): RFS threshold t; classes present in fewer than t of the tiles
            are over-sampled
        max_repeat (float): maximum repeat factor of a tile (limits over-fitting on rare tiles)
        max_ignored_fraction (float): pool tiles whose share of ignored pixels
            (no_data + clouds) is above this are dropped (1.0 = keep all)
    Returns:
        manifest (pandas.DataFrame): columns sample_id, tile_idx (-1 = intact image), kind
        report (pandas.DataFrame): per-class statistics, natural vs. effective distribution
        summary (dict): global numbers (counts of images / tiles)
        effective_counts (numpy.array): class pixel counts seen in one pass over the manifest
    """
    if not 0.0 <= intact_fraction <= 1.0:
        raise ValueError(f"intact_fraction must be in [0, 1], got {intact_fraction}")
    train_ids = np.array(sorted(set(int(i) for i in train_ids)), dtype=np.int64)
    stats = tile_stats[tile_stats['sample_id'].isin(train_ids)]
    missing = set(train_ids.tolist()) - set(stats['sample_id'].unique().tolist())
    if missing:
        raise ValueError(f"{len(missing)} train images have no tile stats, e.g. {sorted(missing)[:5]}: "
                         f"recompute the tile stats CSV")

    # 1-2. draw the intact images
    n_intact = int(round(intact_fraction * len(train_ids)))
    perm = rng.permutation(train_ids)
    intact_ids = np.sort(perm[:n_intact])
    pool_ids = perm[n_intact:]

    # 3. pool tiles
    pool = stats[stats['sample_id'].isin(pool_ids)].sort_values(['sample_id', 'tile_idx'])
    pool = pool.reset_index(drop=True)
    counts = pool[CLASS_COLS].to_numpy(dtype=np.float64)
    ignored_share = counts[:, IGNORED_IDX].sum(1) / counts.sum(1).clip(min=1)
    keep = ignored_share <= max_ignored_fraction
    n_dropped = int((~keep).sum())
    pool, counts = pool[keep].reset_index(drop=True), counts[keep]

    presence = class_presence(counts, min_pixel_fraction)
    class_r, presence_freq = class_repeat_factors(presence, threshold)
    if resampling and len(pool):
        tile_r = tile_repeat_factors(presence, class_r, max_repeat)
        repeats = stochastic_round(tile_r, rng)
    else:
        class_r = np.ones(LCD.N_CLASSES)
        repeats = np.ones(len(pool), dtype=np.int64)

    tiles = pool.loc[np.repeat(np.arange(len(pool)), repeats), ['sample_id', 'tile_idx']]
    manifest = pd.concat([
        pd.DataFrame({'sample_id': intact_ids, 'tile_idx': INTACT_TILE_IDX, 'kind': 'intact'}),
        tiles.assign(kind='tile'),
    ], ignore_index=True)
    manifest['sample_id'] = manifest['sample_id'].astype(np.int64)
    manifest['tile_idx'] = manifest['tile_idx'].astype(np.int64)

    # report: natural distribution (all train images) vs. what one pass over the manifest shows
    img_counts = image_class_counts(stats)
    natural = img_counts.loc[train_ids].to_numpy(dtype=np.float64).sum(0)
    intact_counts = img_counts.loc[intact_ids].to_numpy(dtype=np.float64).sum(0) \
        if len(intact_ids) else np.zeros(LCD.N_CLASSES)
    effective = intact_counts + (counts * repeats[:, None]).sum(0)

    def share(x):
        s = np.zeros_like(x)
        total = x[VALID_IDX].sum()
        if total > 0:
            s[VALID_IDX] = x[VALID_IDX] / total
        return s

    nat_share, eff_share = share(natural), share(effective)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(nat_share > 0, eff_share / nat_share, np.nan)
    report = pd.DataFrame({
        'natural_pixels': natural.astype(np.int64),
        'natural_share': nat_share,
        'effective_pixels': effective.astype(np.int64),
        'effective_share': eff_share,
        'share_ratio': ratio,
        'pool_tiles_with_class': presence.sum(0),
        'pool_tile_frequency': presence_freq,
        'class_repeat_factor': class_r,
        'manifest_tiles_with_class': (presence * repeats[:, None]).sum(0),
    }, index=pd.Index(CLASS_COLS, name='class'))

    summary = {
        'n_train_images': int(len(train_ids)),
        'n_intact_images': int(len(intact_ids)),
        'n_pool_images': int(len(pool_ids)),
        'n_pool_tiles': int(len(pool)),
        'n_dropped_tiles_ignored_classes': n_dropped,
        'n_manifest_tiles': int(repeats.sum()),
        'mean_tile_repeat': float(repeats.mean()) if len(repeats) else 0.0,
        'max_tile_repeat': int(repeats.max()) if len(repeats) else 0,
    }
    return manifest, report, summary, effective
