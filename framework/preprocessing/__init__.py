"""
Optional data preprocessing for training.

Every feature in this package is opt-in from the YAML training config: when the
corresponding section is absent or has `enabled: false`, train.py behaves exactly
as the original benchmark.

Modules:
    config         -- helpers to read optional (nested) config values
    tile_stats     -- per-tile class pixel counts of the training masks (CLI + function)
    sampling       -- per-training manifest: intact images + re-sampled tiles
    class_weights  -- per-class loss weights and per-pixel sample weights
    augmentation   -- TensorFlow-native data augmentation (geometric + photometric)
    spectral       -- spectral indices (NDVI, NDWI) appended as extra input channels
    pipeline       -- builds the tf.data train / validation datasets from the above
"""
