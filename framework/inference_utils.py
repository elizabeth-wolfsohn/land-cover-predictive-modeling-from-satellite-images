"""
Optional inference improvements, used by infer.py when the `prediction` section is
present in the inference config.

vector_mode:
    'argmax'  -- benchmark: class proportions of the argmax mask (hard counts)
    'softmax' -- mean of the per-pixel probabilities over the image (soft proportions).
                 Usually better for the KL divergence metric: proportions are smoother,
                 a rare class gets a small non-zero share instead of exactly 0 (which the
                 KL heavily penalizes when the class is actually present).
tta:
    'none' -- single forward pass
    'flip' -- identity + horizontal flip (2 passes)
    'd4'   -- the 8 symmetries of the square (8 passes)
    Probabilities are mapped back to the original orientation and averaged.
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm

TTA_MODES = ('none', 'flip', 'd4')
VECTOR_MODES = ('argmax', 'softmax')


def _transforms(tta):
    """List of (forward, inverse) functions on NHWC batches."""
    ident = (lambda x: x, lambda x: x)
    hflip = (lambda x: tf.reverse(x, axis=[2]), lambda x: tf.reverse(x, axis=[2]))
    if tta == 'none':
        return [ident]
    if tta == 'flip':
        return [ident, hflip]
    if tta == 'd4':
        out = []
        for flip in (False, True):
            for k in range(4):
                def fwd(x, k=k, flip=flip):
                    x = tf.reverse(x, axis=[2]) if flip else x
                    return tf.image.rot90(x, k)

                def inv(x, k=k, flip=flip):
                    x = tf.image.rot90(x, (4 - k) % 4)
                    return tf.reverse(x, axis=[2]) if flip else x
                out.append((fwd, inv))
        return out
    raise ValueError(f"tta must be one of {TTA_MODES}, got {tta!r}")


def predict_probabilities(model, batch, tta='none'):
    """Per-pixel class probabilities (B, H, W, C), averaged over the TTA transforms."""
    transforms = _transforms(tta)
    acc = None
    for fwd, inv in transforms:
        p = inv(tf.convert_to_tensor(model.predict_on_batch(fwd(batch))))
        acc = p if acc is None else acc + p
    return acc / len(transforms)


def probabilities_to_vectors(probs, mode='argmax'):
    """(B, H, W, C) probabilities -> (B, C) class proportion vectors."""
    num_classes = probs.shape[-1]
    if mode == 'argmax':
        onehot = tf.one_hot(tf.argmax(probs, -1), num_classes, dtype=tf.float32)
        return tf.reduce_mean(onehot, axis=[1, 2])
    if mode == 'softmax':
        v = tf.reduce_mean(tf.cast(probs, tf.float32), axis=[1, 2])
        return v / tf.reduce_sum(v, -1, keepdims=True)
    raise ValueError(f"vector_mode must be one of {VECTOR_MODES}, got {mode!r}")


def predict_as_vectors(model, dataset, vector_mode='argmax', tta='none', steps=None):
    """Class proportion vectors for every sample of a (batched, ordered) dataset.
    Returns numpy.array (N, C)."""
    vectors = []
    for batch in tqdm(dataset, total=steps):
        probs = predict_probabilities(model, batch, tta)
        vectors.append(probabilities_to_vectors(probs, vector_mode).numpy())
    return np.concatenate(vectors, 0)
