"""
Losses for the land cover segmentation model (same as the group's framework/losses.py, branch etienne).

CEKLLoss combines two terms:
  - a pixel-level cross-entropy, weighted per class (the benchmark's proxy task),
  - an image-level KL divergence between the true class proportions and the
    proportions predicted as the mean of the softmax over the image (the challenge metric).

    loss = ce_weight * CE_pixels + kl_weight * KL_proportions
"""
import tensorflow as tf


class CEKLLoss(tf.keras.losses.Loss):
    """Pixel cross-entropy + KL divergence of the class proportions of each image.
    The per-pixel sample weights (class weighting) are applied to the cross-entropy term only."""

    def __init__(self, n_classes, ignored_idx=(0, 1), ce_weight=1.0, kl_weight=1.0,
                 eps=1e-8, name='ce_kl', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = int(n_classes)
        self.ignored_idx = [int(i) for i in ignored_idx]
        self.valid_idx = [i for i in range(self.n_classes) if i not in set(self.ignored_idx)]
        self.ce_weight = float(ce_weight)
        self.kl_weight = float(kl_weight)
        self.eps = float(eps)

    def kl_term(self, y_true, y_pred):
        labels = tf.cast(tf.reshape(y_true, tf.shape(y_pred)[:-1]), tf.int32)
        true = tf.reduce_mean(tf.one_hot(labels, self.n_classes, dtype=y_pred.dtype), axis=[1, 2])
        pred = tf.reduce_mean(y_pred, axis=[1, 2])
        true = tf.gather(true, self.valid_idx, axis=-1)
        pred = tf.gather(pred, self.valid_idx, axis=-1)
        true_sum = tf.reduce_sum(true, -1, keepdims=True)
        true = true / tf.maximum(true_sum, 1e-12)
        pred = pred / tf.maximum(tf.reduce_sum(pred, -1, keepdims=True), 1e-12)
        kl = tf.reduce_sum((true + self.eps) * tf.math.log((true + self.eps) / (pred + self.eps)), -1)
        has_valid = tf.cast(tf.squeeze(true_sum, -1) > 0, kl.dtype)
        return kl * has_valid, has_valid

    def _compute(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
        labels = tf.cast(tf.reshape(y_true, tf.shape(y_pred)[:-1]), tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(labels, y_pred)
        if sample_weight is not None:
            ce = ce * tf.cast(tf.reshape(sample_weight, tf.shape(ce)), ce.dtype)
        ce_term = tf.reduce_mean(ce)
        kl, has_valid = self.kl_term(y_true, y_pred)
        kl_term = tf.reduce_sum(kl) / tf.maximum(tf.reduce_sum(has_valid), 1.0)
        return self.ce_weight * ce_term + self.kl_weight * kl_term

    def call(self, y_true, y_pred):
        return self._compute(y_true, y_pred)

    def __call__(self, y_true, y_pred, sample_weight=None):
        with tf.name_scope(self.name):
            return self._compute(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(n_classes=self.n_classes, ignored_idx=self.ignored_idx,
                      ce_weight=self.ce_weight, kl_weight=self.kl_weight, eps=self.eps)
        return config


def build_loss(loss_type='ce', n_classes=10, ignored_idx=(0, 1), ce_weight=1.0, kl_weight=1.0):
    """Return the loss selected in the config ('ce' = the benchmark loss)."""
    if loss_type == 'ce':
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    if loss_type == 'ce_kl':
        return CEKLLoss(n_classes, ignored_idx, ce_weight=ce_weight, kl_weight=kl_weight)
    raise ValueError(f"unknown loss.type: {loss_type!r} (expected 'ce' or 'ce_kl')")
