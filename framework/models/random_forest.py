"""Pixel-wise Random Forest for land cover (CPU or GPU)."""
import numpy as np
import joblib
from tifffile import TiffFile
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import uniform_filter

from framework.dataset import LandCoverData as LCD


def read_image(path) -> np.ndarray:
    """-> (H, W, 4) float32, normalized by TRAIN_PIXELS_MAX like the baseline."""
    with TiffFile(path) as t:
        img = t.asarray()
    if img.shape[-1] != LCD.N_CHANNELS and img.shape[0] == LCD.N_CHANNELS:
        img = np.transpose(img, (1, 2, 0)) 
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


def extract_features_window(img: np.ndarray, window_size=3) -> np.ndarray:
    """
    (H, W, 4) -> (H, W, 18)

    4 bandes originales
    6 indices spectraux
    4 moyennes locales
    4 écarts-types locaux
    """

    b2, b3, b4, b8 = (img[..., i] for i in range(4))
    bands = [b2, b3, b4, b8]

    # 1. Bandes originales
    features = [img]

    # 2. Indices spectraux
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            nd = (bands[i] - bands[j]) / (
                bands[i] + bands[j] + 1e-6
            )
            features.append(nd[..., None])

    # 3. Statistiques locales
    for band in bands:

        local_mean = uniform_filter(
            band, size=window_size, mode='reflect'
        )

        local_mean_sq = uniform_filter(
            band ** 2, size=window_size, mode='reflect'
        )

        local_std = np.sqrt(
            np.maximum(local_mean_sq - local_mean ** 2, 0)
        )

        features.append(local_mean[..., None])
        features.append(local_std[..., None])

    return np.concatenate(features, axis=-1).astype(np.float32)


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)

    frequencies = counts / counts.sum()

    weights = 1.0 / np.sqrt(frequencies)

    # Normalisation: average weight = 1
    weights = weights / np.average(weights, weights=frequencies)

    return dict(zip(classes.tolist(), weights.tolist()))


class LandCoverRF:
    def __init__(
    self,
    backend,
    n_estimators,
    max_depth,
    min_samples_leaf,
    class_weight,
    max_features,
    n_jobs,
    random_state,
    n_bins,
        ):
        if backend not in ('gpu', 'cpu'):
            raise ValueError(f"backend doit être 'gpu' ou 'cpu', reçu : {backend!r}")

        self.backend = backend
        self.random_state = random_state
        self.n_bins = n_bins

        if backend == 'gpu':
            from cuml.ensemble import RandomForestClassifier as cuRF
            self.model = cuRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_bins=n_bins,  
                class_weight=class_weight,        
                random_state=random_state,
                output_type='numpy',
                verbose=2,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                max_features=max_features,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=2,
            )

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if self.backend == 'gpu':
            # Calculate weights
            class_weights = compute_class_weights(y_idx)

            print("Effectifs:", np.bincount(y_idx))
            print("Class weights used:", class_weights)

            # Data preparation for cuML
            X = np.ascontiguousarray(X, dtype=np.float32)
            y_idx = np.ascontiguousarray(y_idx, dtype=np.int32)

            # Distribute sample weights
            sample_weight = np.ascontiguousarray(
                np.array(
                    [class_weights[c] for c in y_idx],
                    dtype=np.float32
                )
            )

            self.model.fit(
                X,
                y_idx,
                sample_weight=sample_weight
            )

        else:
            self.model.fit(X, y_idx)

        return self

    def predict_image(self, img, stride=1):
        feats = extract_features(img)[::stride, ::stride]
        h, w, f = feats.shape
        flat = np.ascontiguousarray(feats.reshape(h * w, f), dtype=np.float32)
        pred_idx = np.asarray(self.model.predict(flat)).astype(np.int64)
        return self.classes_[pred_idx].reshape(h, w).astype(np.int64)

    def save(self, path):
        joblib.dump({
            'backend': self.backend, 'model': self.model, 'classes_': self.classes_,
            'random_state': self.random_state,
            'n_bins': self.n_bins,
        }, path, compress=3)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.backend, obj.model, obj.classes_ = data['backend'], data['model'], data['classes_']
        obj.random_state = data['random_state']
        obj.n_bins = data['n_bins']
        if obj.backend == 'cpu':
            obj.model.verbose = 0
        return obj