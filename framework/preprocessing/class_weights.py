"""
Per-class loss weights, applied as per-pixel sample weights.

Keras' `class_weight` argument of `fit` does not support segmentation targets
(3+ dimensional), so the weights are gathered per pixel from the mask and passed as
the third element of the dataset tuples: (image, mask, sample_weight).

    w(c) = freq(c) ** (-power)        for valid classes
    w normalized so that the average weight of a pixel is 1: sum_c freq(c) * w(c) = 1
    w(c) = min(w(c), max_weight)
    w(ignored classes) = 0            ("no_data" and "clouds" never contribute to the loss)

power = 0   -> no re-balancing (but ignored classes are still masked out)
power = 0.5 -> inverse square-root frequency (default)
power = 1   -> inverse frequency (very aggressive with this dataset: snow ~ 1e-6)

When tiles are re-sampled (sampling.py), the frequencies should be the *effective* ones
of the manifest, otherwise rare classes are compensated twice: use a lower power then.
"""
import numpy as np
import tensorflow as tf

from framework.dataset import LandCoverData as LCD


def compute_class_weights(class_counts, power=0.5, max_weight=10.0,
                          ignored_idx=tuple(LCD.IGNORED_CLASSES_IDX)):
    """Per-class weights from class pixel counts.

    Args:
        class_counts (array-like[N_CLASSES]): pixel count of every class
        power (float): exponent of the inverse frequency
        max_weight (float): upper bound on a class weight
        ignored_idx (iterable[int]): classes whose weight is set to 0
    Returns:
        numpy.array[float32]: one weight per class
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    valid = np.ones(len(counts), dtype=bool)
    valid[list(ignored_idx)] = False
    valid &= counts > 0
    freq = counts / counts[valid].sum()

    weights = np.zeros(len(counts), dtype=np.float64)
    weights[valid] = freq[valid] ** (-power)
    weights[valid] /= (weights[valid] * freq[valid]).sum()  # average pixel weight = 1
    weights = np.minimum(weights, max_weight)
    weights[list(ignored_idx)] = 0.0
    return weights.astype(np.float32)


def make_sample_weight_fn(class_weight):
    """tf.data map function: (image, mask) -> (image, mask, per-pixel weights).
    Works on single samples (H, W, 1) and on batches (B, H, W, 1), any H and W."""
    cw = tf.constant(class_weight, dtype=tf.float32)

    def add_sample_weights(image, mask):
        weights = tf.gather(cw, tf.cast(tf.squeeze(mask, -1), tf.int32))
        return image, mask, weights

    return add_sample_weights
