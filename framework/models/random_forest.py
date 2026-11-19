"""Pixel-wise Random Forest for land cover."""
import numpy as np
import joblib
from tifffile import TiffFile
from sklearn.ensemble import RandomForestClassifier

from framework.dataset import LandCoverData as LCD


def read_image(path) -> np.ndarray:
    """-> (H, W, 4) float32, normalized by TRAIN_PIXELS_MAX like the baseline."""
    with TiffFile(path) as t:
        img = t.asarray()
    if img.shape[-1] != LCD.N_CHANNELS and img.shape[0] == LCD.N_CHANNELS:
        img = np.transpose(img, (1, 2, 0))   # safety if stored channels-first
    return img.astype(np.float32) / LCD.TRAIN_PIXELS_MAX


def read_mask(path) -> np.ndarray:
    """-> (H, W) int64"""
    with TiffFile(path) as t:
        return t.asarray().astype(np.int64)


def mask_path_from_image(image_path):
    return image_path.parent.parent / 'masks' / image_path.name



def extract_features_window(
    img: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """
    (H, W, 4) -> (H, W, 24)

    Features:
    - 4 bandes spectrales
    - NDVI, NDWI, SAVI, EVI
    - Moyennes locales des 8 variables
    - Écarts-types locaux des 8 variables
    """

    from numpy.lib.stride_tricks import sliding_window_view

    if img.ndim != 3 or img.shape[-1] != 4:
        raise ValueError("img must have shape (H, W, 4)")

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number")

    b2, b3, b4, b8 = (img[..., i] for i in range(4))
    eps = 1e-6

    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    savi = 1.5 * (b8 - b4) / (b8 + b4 + 0.5 + eps)
    evi = 2.5 * (b8 - b4) / (
        b8 + 6 * b4 - 7.5 * b2 + 1 + eps
    )

    variables = np.stack(
        [b2, b3, b4, b8, ndvi, ndwi, savi, evi],
        axis=-1
    ).astype(np.float32)

    radius = window_size // 2

    windows = sliding_window_view(
        variables,
        (window_size, window_size),
        axis=(0, 1)
    )

    local_mean = windows.mean(axis=(-1, -2))
    local_std = windows.std(axis=(-1, -2))

    if radius > 0:
        padding = (
            (radius, radius),
            (radius, radius),
            (0, 0)
        )

        local_mean = np.pad(local_mean, padding, mode="reflect")
        local_std = np.pad(local_std, padding, mode="reflect")

    return np.concatenate(
        [variables, local_mean, local_std],
        axis=-1
    ).astype(np.float32)


# def extract_features(img: np.ndarray) -> np.ndarray:
#     """(H, W, 4) -> (H, W, 10) feature cube."""

#     b2, b3, b4, b8 = (img[..., i] for i in range(4))

#     bands = [b2, b3, b4, b8]

#     features = [img]

#     for i in range(len(bands)):
#         for j in range(i + 1, len(bands)):
#             nd = (bands[i] - bands[j]) / (
#                 bands[i] + bands[j] + 1e-6
#             )
#             features.append(nd[..., None])

#     return np.concatenate(features, axis=-1).astype(np.float32)



def extract_features(img: np.ndarray) -> np.ndarray:
    """
    (H, W, 4) -> (H, W, 8) feature cube.

    Features:
    - B2 (Blue)
    - B3 (Green)
    - B4 (Red)
    - B8 (NIR)
    - NDVI
    - NDWI
    - SAVI
    - EVI
    """

    b2, b3, b4, b8 = (img[..., i] for i in range(4))

    # 1. Original spectral bands
    features = [img]

    # 2. NDVI: vegetation index
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)

    # 3. NDWI: water index (McFeeters)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)

    # 4. SAVI: soil-adjusted vegetation index
    savi = 1.5 * (b8 - b4) / (b8 + b4 + 0.5 + 1e-6)

    # 5. EVI: enhanced vegetation index
    evi = 2.5 * (b8 - b4) / (
        b8 + 6 * b4 - 7.5 * b2 + 1 + 1e-6
    )

    features.extend([
        ndvi[..., None],
        ndwi[..., None],
        savi[..., None],
        evi[..., None]
    ])

    return np.concatenate(features, axis=-1).astype(np.float32)

class LandCoverRF:
    def __init__(
        self,
        n_estimators=50,
        max_depth=25,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        max_features='sqrt',
        n_jobs=2,
        random_state=42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=2
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_image(self, img: np.ndarray, stride: int = 1) -> np.ndarray:
        """(H, W, 4) -> (H/stride, W/stride) predicted class map."""
        feats = extract_features(img)[::stride, ::stride]
        h, w, f = feats.shape
        pred = self.model.predict(feats.reshape(h * w, f))
        return pred.reshape(h, w).astype(np.int64)

    def predict_proportions(self, img: np.ndarray, stride: int = 1) -> np.ndarray:
        """(H, W, 4) -> (10,) class proportions, same as the baseline's vectors."""
        pred = self.predict_image(img, stride)
        counts = np.bincount(pred.ravel(), minlength=LCD.N_CLASSES)
        return counts / counts.sum()

    def save(self, path):
        joblib.dump(self.model, path, compress=3)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = joblib.load(path)
        obj.model.verbose = 0
        return obj