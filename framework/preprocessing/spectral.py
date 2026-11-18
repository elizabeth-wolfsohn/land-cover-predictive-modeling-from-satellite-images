"""
Spectral indices appended to the image as extra input channels.

    NDVI = (NIR - Red)   / (NIR + Red)     vegetation  (high on broadleaf / coniferous / crops)
    NDWI = (Green - NIR) / (Green + NIR)   open water  (McFeeters 1996, high on water)

Both are ratios, so they do not depend on the global rescaling of the pixel values
(dividing all bands by TRAIN_PIXELS_MAX in dataset.normalize): they can be computed
before or after normalization. Values are in [-1, 1].

The band order of the images is assumed to be R, G, B, NIR (`band_order` in the config
can override it). Check it in preprocessing_visualization.ipynb: NIR must be high on
vegetation and very low on water.

The same indices must be computed at inference: train.py saves its config in the
experiment folder and infer.py reads it back.
"""
import tensorflow as tf

from framework.preprocessing.config import cfg_get, section_enabled

DEFAULT_BAND_ORDER = {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}
EPS = 1e-6


def _normalized_difference(a, b):
    return tf.clip_by_value((a - b) / (a + b + EPS), -1.0, 1.0)


INDEX_DEFINITIONS = {
    'ndvi': lambda bands, bo: _normalized_difference(bands[..., bo['nir']], bands[..., bo['red']]),
    'ndwi': lambda bands, bo: _normalized_difference(bands[..., bo['green']], bands[..., bo['nir']]),
}


class SpectralIndices:
    """Computes a list of spectral indices from the (normalized, float) bands."""

    def __init__(self, names=('ndvi', 'ndwi'), band_order=None):
        names = [n.lower() for n in names]
        unknown = [n for n in names if n not in INDEX_DEFINITIONS]
        if unknown:
            raise ValueError(f"unknown spectral indices {unknown}, available: {list(INDEX_DEFINITIONS)}")
        if not names:
            raise ValueError("spectral_indices is enabled but `indices` is empty")
        self.names = names
        self.band_order = dict(DEFAULT_BAND_ORDER, **(band_order or {}))

    @property
    def n_channels(self):
        return len(self.names)

    def compute(self, bands):
        """(..., H, W, C) float bands -> (..., H, W, n_indices) indices."""
        bands = tf.cast(bands, tf.float32)
        return tf.stack([INDEX_DEFINITIONS[n](bands, self.band_order) for n in self.names], axis=-1)

    def append(self, bands):
        """(..., H, W, C) -> (..., H, W, C + n_indices)."""
        bands = tf.cast(bands, tf.float32)
        return tf.concat([bands, self.compute(bands)], axis=-1)

    def __repr__(self):
        return f"SpectralIndices(names={self.names}, band_order={self.band_order})"


def from_config(config):
    """SpectralIndices from the `spectral_indices` config section, or None if disabled."""
    if not section_enabled(config, 'spectral_indices'):
        return None
    band_order = cfg_get(config, 'spectral_indices.band_order', None)
    if band_order is not None and not isinstance(band_order, dict):
        band_order = vars(band_order)  # YamlNamespace -> dict
    return SpectralIndices(
        names=cfg_get(config, 'spectral_indices.indices', ['ndvi', 'ndwi']),
        band_order=band_order,
    )
