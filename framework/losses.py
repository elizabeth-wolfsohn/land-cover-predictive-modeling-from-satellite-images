"""
Losses for the land cover segmentation model (PyTorch port of the group's losses.py, TensorFlow).

CEKLLoss combines two terms:
  - a pixel-level cross-entropy, weighted per class (the benchmark's proxy task),
  - an image-level KL divergence between the true class proportions and the
    proportions predicted as the mean of the softmax over the image (the challenge metric).

    loss = ce_weight * CE_pixels + kl_weight * KL_proportions
"""
import torch
import torch.nn.functional as F


class CEKLLoss(torch.nn.Module):
    """Pixel cross-entropy + KL divergence of the class proportions of each image.

    Args:
        n_classes (int): number of classes (output channels of the model)
        ignored_idx (list[int]): classes removed from the KL term (no_data, clouds),
            as done by the challenge metric
        ce_weight (float): weight of the cross-entropy term
        kl_weight (float): weight of the KL term
        eps (float): smoothing constant of the KL (the challenge uses 1e-8)
        class_weight (torch.Tensor | None): (n_classes,) per-class weights of the cross-entropy

    Unlike the TensorFlow version, the model output is expected as logits (B, C, H, W) and the
    class weights are given here (Keras passed them as per-pixel sample weights). As with Keras'
    sample weights, the weighted cross-entropy is averaged over all the pixels (not divided by the
    sum of the weights like torch.nn.CrossEntropyLoss(weight=...)). The class weights are applied
    to the cross-entropy term only: the KL term is computed on unweighted proportions, so that it
    stays an estimate of the challenge metric.
    """

    def __init__(self, n_classes, ignored_idx=(0, 1), ce_weight=1.0, kl_weight=1.0,
                 eps=1e-8, class_weight=None):
        super().__init__()
        self.n_classes = int(n_classes)
        self.ignored_idx = [int(i) for i in ignored_idx]
        self.valid_idx = [i for i in range(self.n_classes) if i not in set(self.ignored_idx)]
        self.ce_weight = float(ce_weight)
        self.kl_weight = float(kl_weight)
        self.eps = float(eps)
        if class_weight is not None:
            class_weight = torch.as_tensor(class_weight, dtype=torch.float32)
            assert class_weight.shape == (self.n_classes,), class_weight.shape
        self.register_buffer('class_weight', class_weight)

    def kl_term(self, logits, labels):
        """KL(true proportions || predicted proportions) for every image, shape (B,),
        and a (B,) mask of the images that have evaluated pixels.

        Images without any evaluated pixel (only no_data / clouds) get a KL of 0.
        """
        pred = logits.softmax(1).mean((2, 3))                                      # (B, C)
        true = F.one_hot(labels, self.n_classes).to(pred.dtype).mean((1, 2))      # (B, C)
        true, pred = true[:, self.valid_idx], pred[:, self.valid_idx]
        true_sum = true.sum(-1, keepdim=True)
        true = true / true_sum.clamp_min(1e-12)
        pred = pred / pred.sum(-1, keepdim=True).clamp_min(1e-12)
        kl = ((true + self.eps) * torch.log((true + self.eps) / (pred + self.eps))).sum(-1)
        has_valid = (true_sum.squeeze(-1) > 0).to(kl.dtype)
        return kl * has_valid, has_valid

    def forward(self, logits, labels):
        """logits: (B, C, H, W), labels: (B, H, W) integer class indices."""
        logits = logits.float()
        labels = labels.long()

        # 1. pixel cross-entropy, weighted by the per-pixel class weights
        ce = F.cross_entropy(logits, labels, reduction='none')                     # (B, H, W)
        if self.class_weight is not None:
            ce = ce * self.class_weight[labels]
        ce_term = ce.mean()    # same normalization as Keras' default reduction

        # 2. KL of the proportions, averaged over the images that have evaluated pixels
        kl, has_valid = self.kl_term(logits, labels)
        kl_term = kl.sum() / has_valid.sum().clamp_min(1.0)

        return self.ce_weight * ce_term + self.kl_weight * kl_term


def build_loss(loss_type='ce', n_classes=10, ignored_idx=(0, 1), ce_weight=1.0, kl_weight=1.0,
               class_weight=None):
    """Return the loss selected in the config ('ce' = the benchmark loss)."""
    if loss_type == 'ce':
        return torch.nn.CrossEntropyLoss(weight=class_weight)
    if loss_type == 'ce_kl':
        return CEKLLoss(n_classes, ignored_idx, ce_weight=ce_weight, kl_weight=kl_weight,
                        class_weight=class_weight)
    raise ValueError(f"loss.type inconnu : {loss_type!r} (attendu : 'ce' ou 'ce_kl')")
