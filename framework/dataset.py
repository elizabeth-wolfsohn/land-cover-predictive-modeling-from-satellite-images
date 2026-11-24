"""
Classes and functions to handle data
(PyTorch port of dataset_tf.py, which is the reference)
"""
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
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

    # The original training dataset (18491 images and masks) is split into
    # train / validation / test (train_images.csv, validation_images.csv, test_images.csv)
    TRAINSET_SIZE = 14794  # 18491 initially
    VALSET_SIZE = 1848
    TESTSET_SIZE = 1849  # 5043 initially

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
        [0, 17441, 48775625, 323881666, 222414642, 76948228, 265937117, 7749139, 828, 23814898]
        # [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605] #initially
    )
    # the pixel class counts in the validation set
    VAL_CLASS_COUNTS = np.array(
        [0, 1249, 6258733, 40140226, 27516024, 10176292, 33510997, 1030736, 23, 2476248]
    )

    # the minimum and maximum value of image pixels in the training set
    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356


def files_from_csv(csv_path, images_dir):
    """List of image paths `<images_dir>/<sample_id>.tif` from a split csv
    (train_images.csv, validation_images.csv, test_images.csv, or a *_labels.csv)."""
    ids = pd.read_csv(csv_path)['sample_id']
    return [Path(images_dir)/f'{i}.tif' for i in ids]


def input_normalization(config):
    """(mean, std) used to normalize the images: as in the reference, every model uses
    image / TRAIN_PIXELS_MAX (kept to not break the train script, config is ignored)."""
    return (np.zeros(LandCoverData.N_CHANNELS, dtype='float32'),
            np.full(LandCoverData.N_CHANNELS, LandCoverData.TRAIN_PIXELS_MAX, dtype='float32'))


class LandCoverDataset(torch.utils.data.Dataset):
    """Images (and masks) from a list of `<root>/images/<id>.tif` files, as in dataset_tf.py:
    channels in the .tif order, image / TRAIN_PIXELS_MAX, random flip then random 90° rotation.
    Returns (image[4,256,256] float32, mask[256,256,1] uint8) or the image only.
    norm: (mean, std) of the normalization, (0, TRAIN_PIXELS_MAX) like the reference by default.
    """
    def __init__(self, files, augment=False, with_mask=True, norm=None):
        self.files, self.augment, self.with_mask = list(files), augment, with_mask
        self.mean, self.std = input_normalization(None) if norm is None else norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # numpy_parse_image: image (256, 256, 4) uint16, mask (256, 256, 1) uint8
        image_path = Path(self.files[idx])
        image = imread(image_path).astype('uint16')
        if not self.with_mask:
            image = self._normalize(image)
            return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # the mask is at masks/<image_id>.tif next to images/<image_id>.tif
        mask = imread(image_path.parent.parent/'masks'/image_path.name).astype('uint8')
        mask = mask[..., None]
        if self.augment:
            # load_image_train: random left-right flip, then rotation by k*90° counter-clockwise
            # (tf.image.flip_left_right / tf.image.rot90 on (H, W, C) arrays)
            if np.random.rand() > 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            k = np.random.randint(4)
            image, mask = np.rot90(image, k, (0, 1)), np.rot90(mask, k, (0, 1))
        image = self._normalize(image)
        image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W) for PyTorch
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(mask))

    def _normalize(self, image):
        """normalize: rescale the pixel values of the images (cast to float32 / TRAIN_PIXELS_MAX).
        Inside a tf.function, Grappler rewrites `x / c` into `x * (1/c)` (float32): multiplying by
        the float32 inverse gives bit-identical values (a true division differs by 1 ulp)."""
        inv_std = np.float32(1) / np.asarray(self.std, dtype='float32')
        return (image.astype('float32') - self.mean) * inv_std


def class_distribution(x, n_classes=LandCoverData.N_CLASSES):
    """Per-image class distribution vectors.
    Args:
        x (torch.Tensor): integer masks (B, H, W) or (B, H, W, 1) of class indices,
            or float probabilities (B, C, H, W)
    Returns:
        (torch.Tensor): (B, n_classes) vectors summing to 1
    """
    if x.is_floating_point():
        return x.float().mean((2, 3))
    counts = torch.stack([torch.bincount(m.flatten().long(), minlength=n_classes) for m in x])
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