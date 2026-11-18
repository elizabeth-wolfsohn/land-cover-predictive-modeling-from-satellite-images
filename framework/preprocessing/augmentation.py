"""
TensorFlow-native data augmentation for multispectral (R, G, B, NIR) segmentation.

Why native TensorFlow rather than albumentations: albumentations works on numpy arrays,
so it would need a tf.numpy_function in the tf.data pipeline (Python GIL, no graph
optimization, CPU bottleneck), and most of its color transforms assume 3-channel
uint8 RGB images, not 4-channel uint16 reflectances. Every transform below runs inside
the tf.data graph and works with any number of channels and any image size (256 or 128).

Two families:
  * geometric   -- applied identically to the image and the mask
      - flip + rot90: the 8 symmetries of the square (no interpolation, labels exact)
      - zoom: crop a random square sub-region (side in [min_scale, 1] of the image)
              and resize it back (bilinear for the image, nearest for the mask)
  * photometric -- applied to the spectral bands only (never to the mask, and computed
                   *after* the spectral indices, which stay physically consistent)
      - brightness: x * b,                                b ~ U(range)
      - contrast:   mean_c + (x - mean_c) * k,            k ~ U(range), per-channel mean
      - color:      x * g_c, one gain per band,           g_c ~ U(range)  (spectral shift)
      - noise:      x + N(0, std * mean_c)                (relative to the band level)
    All photometric transforms are multiplicative / relative, so they do not depend on
    the pixel scale (images are divided by TRAIN_PIXELS_MAX, values ~ 0.0 - 0.2).

Config (section `augmentation`, every key optional):
    augmentation:
      enabled: true
      flip_rot90: true
      zoom:       {prob: 0.3, min_scale: 0.6}
      brightness: {prob: 0.5, range: [0.9, 1.1]}
      contrast:   {prob: 0.5, range: [0.85, 1.15]}
      color:      {prob: 0.5, range: [0.93, 1.07]}
      noise:      {prob: 0.0, std: 0.02}
"""
import tensorflow as tf

from framework.preprocessing.config import cfg_get

DEFAULTS = {
    'flip_rot90': True,
    'zoom': {'prob': 0.3, 'min_scale': 0.6},
    'brightness': {'prob': 0.5, 'range': [0.9, 1.1]},
    'contrast': {'prob': 0.5, 'range': [0.85, 1.15]},
    'color': {'prob': 0.5, 'range': [0.93, 1.07]},
    'noise': {'prob': 0.0, 'std': 0.02},
}


# ---------------------------------------------------------------------------
# geometric transforms: (image, mask) -> (image, mask), shapes (H, W, C) / (H, W, 1)
# ---------------------------------------------------------------------------

def random_flip_rot90(image, mask):
    """One of the 8 symmetries of the square, uniformly (flip x rot90 k)."""
    flip = tf.random.uniform(()) < 0.5
    image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(image, k), tf.image.rot90(mask, k)


def random_zoom(image, mask, min_scale=0.6, max_scale=1.0):
    """Crop a random square-ratio sub-region and resize it back to the input size."""
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    scale = tf.random.uniform((), min_scale, max_scale)
    ch = tf.maximum(1, tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32))
    cw = tf.maximum(1, tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32))
    oy = tf.random.uniform((), 0, h - ch + 1, dtype=tf.int32)
    ox = tf.random.uniform((), 0, w - cw + 1, dtype=tf.int32)
    img = tf.image.resize(image[oy:oy + ch, ox:ox + cw, :], [h, w], method='bilinear')
    msk = tf.image.resize(mask[oy:oy + ch, ox:ox + cw, :], [h, w], method='nearest')
    img.set_shape(image.shape)
    msk.set_shape(mask.shape)
    return tf.cast(img, image.dtype), tf.cast(msk, mask.dtype)


# ---------------------------------------------------------------------------
# photometric transforms: bands (H, W, C) float -> bands
# ---------------------------------------------------------------------------

def random_brightness(x, low=0.9, high=1.1):
    return x * tf.random.uniform((), low, high)


def random_contrast(x, low=0.85, high=1.15):
    mean = tf.reduce_mean(x, axis=[-3, -2], keepdims=True)
    return mean + (x - mean) * tf.random.uniform((), low, high)


def random_color(x, low=0.93, high=1.07):
    gains = tf.random.uniform([tf.shape(x)[-1]], low, high)
    return x * gains


def gaussian_noise(x, std=0.02):
    mean = tf.reduce_mean(x, axis=[-3, -2], keepdims=True)
    return x + tf.random.normal(tf.shape(x)) * std * mean


# ---------------------------------------------------------------------------

def _maybe(prob, fn, *tensors):
    """Apply fn(*tensors) with probability prob; returns a tuple, static shapes kept."""
    if prob <= 0:
        return tensors
    if prob >= 1:
        out = fn(*tensors)
    else:
        out = tf.cond(tf.random.uniform(()) < prob,
                      lambda: tuple(fn(*tensors)) if len(tensors) > 1 else (fn(*tensors),),
                      lambda: tensors)
    out = tuple(out) if isinstance(out, (tuple, list)) else (out,)
    for o, t in zip(out, tensors):
        o.set_shape(t.shape)
    return out


class Augmenter:
    """Configurable augmentation pipeline.

    Usage:
        aug = Augmenter.from_config(config)          # None if disabled
        image, mask = aug.geometric(image, mask)     # float image, uint8 mask
        bands = aug.photometric(bands)               # spectral bands only
    """

    def __init__(self, params=None):
        p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
        for key, value in (params or {}).items():
            if isinstance(value, dict) and isinstance(p.get(key), dict):
                p[key].update(value)
            else:
                p[key] = value
        self.params = p

    @classmethod
    def from_config(cls, config):
        """Augmenter from the `augmentation` section, or None if absent / disabled."""
        if not cfg_get(config, 'augmentation.enabled', False):
            return None
        params = {}
        for key, default in DEFAULTS.items():
            if isinstance(default, dict):
                params[key] = {k: cfg_get(config, f'augmentation.{key}.{k}', v) for k, v in default.items()}
            else:
                params[key] = cfg_get(config, f'augmentation.{key}', default)
        return cls(params)

    def geometric(self, image, mask):
        p = self.params
        if p['flip_rot90']:
            image, mask = random_flip_rot90(image, mask)
        z = p['zoom']
        image, mask = _maybe(float(z['prob']), lambda i, m: random_zoom(i, m, float(z['min_scale'])),
                             image, mask)
        return image, mask

    def photometric(self, bands):
        p = self.params
        b, c, col, n = p['brightness'], p['contrast'], p['color'], p['noise']
        bands, = _maybe(float(b['prob']), lambda x: random_brightness(x, *map(float, b['range'])), bands)
        bands, = _maybe(float(c['prob']), lambda x: random_contrast(x, *map(float, c['range'])), bands)
        bands, = _maybe(float(col['prob']), lambda x: random_color(x, *map(float, col['range'])), bands)
        bands, = _maybe(float(n['prob']), lambda x: gaussian_noise(x, float(n['std'])), bands)
        return tf.maximum(bands, 0.0)

    def __call__(self, image, mask):
        """Geometric then photometric (convenience, e.g. for visualization)."""
        image, mask = self.geometric(image, mask)
        return self.photometric(image), mask

    def __repr__(self):
        return f"Augmenter({self.params})"
