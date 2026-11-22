"""
Keras metrics computed during training on segmentation outputs.
The classes "no_data" (0) and "clouds" (1) are ignored, as in the challenge metric.
"""
import tensorflow as tf

from framework.dataset import LandCoverData as LCD

EVAL_CLASSES = [c for c in range(LCD.N_CLASSES) if c not in LCD.IGNORED_CLASSES_IDX]


def _labels(y_true, y_pred):
    """(B, H, W[, 1]) labels -> (B, H, W) int32"""
    return tf.cast(tf.reshape(y_true, tf.shape(y_pred)[:-1]), tf.int32)


class PixelAccuracy(tf.keras.metrics.Metric):
    """Fraction of correctly classified pixels, ignoring the "no_data" and "clouds" pixels.
    Note: with a strong class imbalance, this metric is dominated by the frequent classes."""

    def __init__(self, name='accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros', dtype=tf.float64)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = _labels(y_true, y_pred)
        pred = tf.cast(tf.argmax(y_pred, -1), tf.int32)
        valid = tf.reduce_all(
            tf.not_equal(labels[..., None], tf.constant(LCD.IGNORED_CLASSES_IDX, tf.int32)), -1)
        correct = tf.logical_and(valid, tf.equal(labels, pred))
        self.correct.assign_add(tf.reduce_sum(tf.cast(correct, tf.float64)))
        self.total.assign_add(tf.reduce_sum(tf.cast(valid, tf.float64)))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.)
        self.total.assign(0.)


class ClassDistributionKL(tf.keras.metrics.Metric):
    """The challenge metric: mean KL divergence between the true class distribution of every
    image and the predicted one (average of the pixel probabilities = "soft" aggregation)."""

    def __init__(self, name='kl', epsilon=1e-7, **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.kl_sum = self.add_weight(name='kl_sum', initializer='zeros', dtype=tf.float64)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = _labels(y_true, y_pred)
        eval_idx = tf.constant(EVAL_CLASSES, tf.int32)
        true_counts = tf.reduce_sum(tf.one_hot(labels, LCD.N_CLASSES, dtype=tf.float64), axis=(1, 2))
        p_true = tf.gather(true_counts, eval_idx, axis=-1)
        p_pred = tf.gather(tf.reduce_mean(tf.cast(y_pred, tf.float64), axis=(1, 2)), eval_idx, axis=-1)
        p_true = tf.math.divide_no_nan(p_true, tf.reduce_sum(p_true, -1, keepdims=True))
        p_pred = tf.math.divide_no_nan(p_pred, tf.reduce_sum(p_pred, -1, keepdims=True))
        eps = tf.constant(self.epsilon, tf.float64)
        kl = tf.reduce_sum((p_true + eps) * tf.math.log((p_true + eps) / (p_pred + eps)), -1)
        self.kl_sum.assign_add(tf.reduce_sum(kl))
        self.count.assign_add(tf.cast(tf.shape(kl)[0], tf.float64))

    def result(self):
        return tf.math.divide_no_nan(self.kl_sum, self.count)

    def reset_state(self):
        self.kl_sum.assign(0.)
        self.count.assign(0.)

    def get_config(self):
        return {**super().get_config(), 'epsilon': self.epsilon}
