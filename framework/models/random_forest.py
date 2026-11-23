"""Pixel-wise Random Forest for land cover (CPU or GPU)."""
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




def balance_pixels(X, y, max_per_class=200_000, seed=42):
    """Remplace class_weight='balanced_subsample' (non supporté par cuML) :
    sous-échantillonne les classes majoritaires."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        if len(ci) > max_per_class:
            ci = rng.choice(ci, max_per_class, replace=False)
        idx.append(ci)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx] 



class LandCoverRF:
    def __init__(
        self,
        backend='gpu',               # 'gpu' (cuML) ou 'cpu' (scikit-learn)
        n_estimators=50,
        max_depth=25,
        min_samples_leaf=5,
        class_weight='balanced_subsample',   # CPU uniquement
        max_features='sqrt',
        n_jobs=2,                            # CPU uniquement
        random_state=42,
    ):
        self.backend = backend
        if backend == 'gpu':
            from cuml.ensemble import RandomForestClassifier as cuRF
            self.model = cuRF(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_bins=128,
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
        self.classes_, y_idx = np.unique(y, return_inverse=True)   # classes_ = [2..9]
        if self.backend == 'gpu':
            X, y_idx = balance_pixels(X, y_idx)
            X = np.ascontiguousarray(X, dtype=np.float32)
            y_idx = np.ascontiguousarray(y_idx, dtype=np.int32)
        self.model.fit(X, y_idx)
        return self

    def predict_image(self, img, stride=1):
        feats = extract_features(img)[::stride, ::stride]
        h, w, f = feats.shape
        flat = np.ascontiguousarray(feats.reshape(h * w, f), dtype=np.float32)
        pred_idx = np.asarray(self.model.predict(flat)).astype(np.int64)
        return self.classes_[pred_idx].reshape(h, w).astype(np.int64)

    def save(self, path):
        joblib.dump({'backend': self.backend, 'model': self.model,
                    'classes_': self.classes_}, path, compress=3)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.backend, obj.model, obj.classes_ = data['backend'], data['model'], data['classes_']
        if obj.backend == 'cpu':
            obj.model.verbose = 0
        return obj