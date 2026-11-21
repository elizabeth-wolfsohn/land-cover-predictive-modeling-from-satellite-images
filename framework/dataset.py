"""
Classes and functions to handle data
"""
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from tifffile import imread


class LandCoverData():
    """Class to represent the S2GLC Land Cover Dataset for the challenge,
    with useful metadata and statistics.
    """
    # image size of the images and label masks
    IMG_SIZE = 256
    # the images are RGB+NIR (4 channels)
    N_CHANNELS = 4
    # we have 9 classes + a 'no_data' class for pixels with no labels (absent in the dataset)
    N_CLASSES = 10
    CLASSES = [
        'no_data',
        'clouds',
        'artificial',
        'cultivated',
        'broadleaf',
        'coniferous',
        'herbaceous',
        'natural',
        'snow',
        'water'
    ]
    # classes to ignore because they are not relevant. "no_data" refers to pixels without
    # a proper class, but it is absent in the dataset; "clouds" class is not relevant, it
    # is not a proper land cover type and images and masks do not exactly match in time.
    IGNORED_CLASSES_IDX = [0, 1]

    # The training dataset contains 18491 images and masks
    # The test dataset contains 5043 images and masks
    TRAINSET_SIZE = 18491
    TESTSET_SIZE = 5043

    # for visualization of the masks: classes indices and RGB colors
    CLASSES_COLORPALETTE = {
        0: [0,0,0],
        1: [255,25,236],
        2: [215,25,28],
        3: [211,154,92],
        4: [33,115,55],
        5: [21,75,35],
        6: [118,209,93],
        7: [130,130,130],
        8: [255,255,255],
        9: [43,61,255]
        }
    CLASSES_COLORPALETTE = {c: np.asarray(color) for (c, color) in CLASSES_COLORPALETTE.items()}

    # statistics
    # the pixel class counts in the training set
    TRAIN_CLASS_COUNTS = np.array(
        [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605]
    )
    # the minimum and maximum value of image pixels in the training set
    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356
    # the .tif files store the bands as B,G,R,NIR: reorder them as R,G,B,NIR when loading,
    # to match the RGB order of ImageNet pretrained weights
    CHANNEL_ORDER = [2, 1, 0, 3]
    # per-channel (R,G,B,NIR) mean and std of image pixels (computed on 1000 random training images)
    CHANNEL_MEAN = np.array([651.4, 695.4, 411.2, 3282.4], dtype='float32')
    CHANNEL_STD = np.array([592.8, 427.0, 367.1, 1006.5], dtype='float32')
    # SatlasPretrain inputs: L1C TCI / 255 for R,G,B (TCI saturates at 3558) and B08 / 8160
    SATLAS_SCALE = np.array([3558., 3558., 3558., 8160.], dtype='float32')


def input_normalization(config):
    """(mean, std) used to normalize the images for the model of a train config."""
    if config.model == 'swin' and config.backbone.startswith('satlas'):
        # ponytail: no clipping to [0, 1] like Satlas, bright clouds/snow pixels go above 1
        return np.zeros(4, dtype='float32'), LandCoverData.SATLAS_SCALE
    return LandCoverData.CHANNEL_MEAN, LandCoverData.CHANNEL_STD


class LandCoverDataset(torch.utils.data.Dataset):
    """Images (and masks) from a list of `<root>/images/<id>.tif` files.
    Returns (image[4,256,256] float32 normalized R,G,B,NIR, mask[256,256] int64) or the image only.
    norm: (mean, std) of the normalization, per-channel standardization by default.
    """
    def __init__(self, files, augment=False, with_mask=True,
                 norm=(LandCoverData.CHANNEL_MEAN, LandCoverData.CHANNEL_STD)):
        self.files, self.augment, self.with_mask = list(files), augment, with_mask
        self.mean, self.std = norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = Path(self.files[idx])
        image = imread(image_path)[..., LandCoverData.CHANNEL_ORDER].astype('float32')
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)  # (C, H, W)
        if not self.with_mask:
            return torch.from_numpy(np.ascontiguousarray(image))
        # the mask is at masks/<image_id>.tif next to images/<image_id>.tif
        mask = imread(image_path.parent.parent/'masks'/image_path.name).astype('int64')
        if self.augment:
            # random flips and 90 degrees rotations (the 8 dihedral transforms)
            k = np.random.randint(4)
            image, mask = np.rot90(image, k, (1, 2)), np.rot90(mask, k)
            if np.random.rand() > 0.5:
                image, mask = image[:, :, ::-1], mask[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(mask))


def class_distribution(x, n_classes=LandCoverData.N_CLASSES):
    """Per-image class distribution vectors.
    Args:
        x (torch.Tensor): masks (B, H, W) of class indices, or probabilities (B, C, H, W)
    Returns:
        (torch.Tensor): (B, n_classes) vectors summing to 1
    """
    if x.ndim == 4:
        return x.float().mean((2, 3))
    counts = torch.stack([torch.bincount(m.flatten(), minlength=n_classes) for m in x])
    return counts.float() / counts.sum(1, keepdim=True)


def predict_distribution(logits, mode='argmax'):
    """Image-level class distribution predicted from (B, C, H, W) logits.
    The ignored classes ("no_data", "clouds") are never predicted.
    mode: 'argmax' counts the most confident class of every pixel, 'soft' averages the probabilities.
    """
    logits = logits.float().clone()
    logits[:, LandCoverData.IGNORED_CLASSES_IDX] = -torch.inf
    if mode == 'soft':
        return class_distribution(logits.softmax(1))
    return class_distribution(logits.argmax(1), logits.shape[1])
