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


def extract_features(img: np.ndarray) -> np.ndarray:
    """(H, W, 4) -> (H, W, 10) feature cube."""

    b2, b3, b4, b8 = (img[..., i] for i in range(4))

    bands = [b2, b3, b4, b8]

    features = [img]

    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            nd = (bands[i] - bands[j]) / (
                bands[i] + bands[j] + 1e-6
            )
            features.append(nd[..., None])

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