"""
Evaluate model predictions:
- KL divergence using class-proportion CSV files.
- Pixel-wise confusion matrix using TIFF masks.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import yaml

from tifffile import TiffFile

from framework.dataset import LandCoverData as LCD

DATASET_FOLDER_PATH = "challenge-ens/data/dataset_UNZIPPED"

def epsilon_kl_divergence(y_true, y_pred):
    """
        The metric of the challenge, the Kullback-Leibler divergence between two discrete distributions.
        Modification includes adding a small constant epsilon (1e-7) to y_true and y_pred
    Args:
            y_true (numpy.array[float]): ground-truth
            y_pred (numpy.arra[float]): prediction
    Returns:
        (float): the score (lower is better)
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)

    # Normalize to sum to 1 if it's not already
    y_true /= y_true.sum(1, keepdims=True)
    y_pred /= y_pred.sum(1, keepdims=True)
    # add a small constant for smoothness around 0
    y_true += 1e-7
    y_pred += 1e-7
    score = np.mean(np.sum(y_true * np.log(y_true / y_pred), 1))
    try:
        assert np.isfinite(score)
    except AssertionError as e:
        raise ValueError('score is NaN or infinite') from e
    return score


def read_image_ids(csv_path):
    """Return the sample IDs listed in the CSV of a set, in the file order.

    The CSV has one column of IDs (e.g. "sample_id"), with or without a header line.
    """
    col = pd.read_csv(csv_path, header=None, dtype=str).iloc[:, 0].str.strip()
    ids = pd.to_numeric(col, errors="coerce")
    # only the first line may be non-numeric (header)
    if ids.iloc[1:].isna().any():
        raise ValueError(f"{csv_path}: non-numeric IDs: {col[ids.isna()].tolist()[:10]}")
    ids = ids.dropna().astype(np.int64)
    if ids.duplicated().any():
        raise ValueError(f"{csv_path}: duplicated IDs: {ids[ids.duplicated()].tolist()[:10]}")
    return ids.tolist()


def _files_by_id(folder):
    """Map sample ID -> TIFF file in a folder (works with or without zero padding)."""
    return {int(f.stem): f for f in Path(folder).glob("*.tif") if f.stem.isdigit()}


def build_confusion_matrix(gt_mask_dir, pred_mask_dir, image_ids):
    """Build a pixel-wise confusion matrix with accuracy and F1-score.

    Only the masks of `image_ids` (the images of the evaluated set) are read in
    gt_mask_dir, in the order of the list. The prediction masks must have the same names.
    """

    n_classes = LCD.N_CLASSES
    class_ids = np.arange(n_classes)

    gt_files = _files_by_id(gt_mask_dir)
    pred_files = _files_by_id(pred_mask_dir)

    if not gt_files:
        raise ValueError(f"No TIFF masks found in {gt_mask_dir}")

    missing_gt = [i for i in image_ids if i not in gt_files]
    if missing_gt:
        raise FileNotFoundError(
            f"{len(missing_gt)} ground-truth mask(s) not found in {gt_mask_dir}, "
            f"e.g. {missing_gt[:10]}"
        )
    missing_pred = [i for i in image_ids if i not in pred_files]
    if missing_pred:
        raise FileNotFoundError(
            f"{len(missing_pred)} prediction mask(s) not found in {pred_mask_dir}, "
            f"e.g. {missing_pred[:10]}"
        )

    # The matrix is accumulated image by image: the memory stays constant,
    # even on the whole training set
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for sample_id in image_ids:
        with TiffFile(gt_files[sample_id]) as t:
            y_true = t.asarray().ravel().astype(np.int64)

        with TiffFile(pred_files[sample_id]) as t:
            y_pred = t.asarray().ravel().astype(np.int64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Different mask dimensions for {gt_files[sample_id].name}: "
                f"{y_true.shape} vs {y_pred.shape}"
            )

        # Ignore pixels belonging to ignored ground-truth classes
        valid = ~np.isin(y_true, LCD.IGNORED_CLASSES_IDX)
        y_true, y_pred = y_true[valid], y_pred[valid]

        # Check class IDs
        if not np.all(np.isin(y_true, class_ids)):
            raise ValueError(f"Unknown class IDs in ground-truth mask {sample_id}.")

        if not np.all(np.isin(y_pred, class_ids)):
            raise ValueError(f"Unknown class IDs in predicted mask {sample_id}.")

        cm += np.bincount(
            y_true * n_classes + y_pred, minlength=n_classes ** 2
        ).reshape(n_classes, n_classes)

    print(f"Confusion matrix: {len(image_ids)} images, {cm.sum()} evaluated pixels")

    # True positives, false positives and false negatives
    tp = np.diag(cm)
    total_true = cm.sum(axis=1)
    total_pred = cm.sum(axis=0)

    # Producer's Accuracy (Recall)
    recall = np.divide(
        tp,
        total_true,
        out=np.zeros(len(class_ids), dtype=float),
        where=total_true != 0
    )

    # User's Accuracy (Precision)
    precision = np.divide(
        tp,
        total_pred,
        out=np.zeros(len(class_ids), dtype=float),
        where=total_pred != 0
    )

    # F1-score
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(class_ids), dtype=float),
        where=(precision + recall) != 0
    )

    iou = np.divide(
        tp,
        total_true + total_pred - tp,
        out=np.zeros(len(class_ids), dtype=float),
        where=(total_true + total_pred - tp) != 0
    )

    # Final table
    df_cm = pd.DataFrame(
        cm,
        index=LCD.CLASSES,
        columns=LCD.CLASSES
    )
    # Remove the first two class columns
    df_cm = df_cm.iloc[2:, :]

    df_cm.index.name = "Ground Truth Class"

        # Add metrics for the remaining classes
    df_cm["Recall (%)"] = recall[2:] * 100
    df_cm["Precision (%)"] = precision[2:] * 100
    df_cm["F1-score (%)"] = f1[2:] * 100
    df_cm["IoU (%)"] = iou[2:] * 100


    return df_cm



def parse_args():
    parser = argparse.ArgumentParser("Evaluation script")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="YAML configuration file"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Load configuration
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_folder = Path(cfg["dataset_folder"]).expanduser()
    xp_rootdir = Path(cfg["xp_rootdir"]).expanduser()

    set_name = cfg["set"]
    
    if set_name not in ("train", "val", "test"):
        raise ValueError("set must be train, val or test.")
    

    ckpt = str(cfg["checkpoint_epoch"])          # 'best' ou un entier
    if ckpt == "best":
        tag = "best"  
    elif ckpt == "rf" :
        tag = "rf"
    else :
        tag=f"epoch{int(ckpt)}"
    prediction_file_name = f"{tag}_{set_name}_predicted.csv"
    


    # Resolve experiment directory
    if cfg['xp_name'] == 'last':
        experiments = [
            d for d in xp_rootdir.iterdir()
            if d.is_dir() and (d / prediction_file_name).is_file()
        ]

        if not experiments:
            raise FileNotFoundError(
                f"Aucun modèle {prediction_file_name} trouvé dans {xp_rootdir}"
            )

        xp_dir = max(experiments, key=lambda d: d.stat().st_mtime)

        print(f"Dernière expérience sélectionnée : {xp_dir.name}")

    else:
        xp_dir = xp_rootdir / cfg['xp_name']

    # Map short set names to dataset folder names
    dataset_set = "validation" if set_name == "val" else set_name

    # Automatically construct file paths
    gt_file = (
        dataset_folder
        / "csv_files"
        / f"{dataset_set}_labels.csv"
    )

    gt_images = (
        dataset_folder
        / "csv_files"
        / f"{dataset_set}_images.csv"
    )


    pred_file = xp_dir / prediction_file_name
    pred_mask_dir = xp_dir / f'{tag}_{set_name}_predicted'

    gt_mask_dir = Path(DATASET_FOLDER_PATH).expanduser()/"train/masks"

    #gt_mask_dir = dataset_folder / dataset_set / "masks"
    #pred_mask_dir = xp_dir / f"{set_name}_predicted_masks"

    out_csv = xp_dir / f"eval_{tag}_{set_name}_results.csv"
    confusion_out = xp_dir / f"confusion_{tag}_{set_name}.csv"
    # Check input files and directories
    if not gt_file.is_file():
        raise FileNotFoundError(gt_file)

    if not gt_images.is_file():
        raise FileNotFoundError(gt_images)

    if not pred_file.is_file():
        raise FileNotFoundError(pred_file)

    if not gt_mask_dir.is_dir():
        raise FileNotFoundError(gt_mask_dir)

    if not pred_mask_dir.is_dir():
        raise FileNotFoundError(pred_mask_dir)

    # --------------------------------------------------
    # 1. KL divergence
    # --------------------------------------------------

    df_y_true = pd.read_csv(gt_file, index_col=0)
    df_y_pred = pd.read_csv(pred_file, index_col=0)

    if not df_y_pred.index.isin(df_y_true.index).all():
        raise ValueError(
            "Some sample IDs in predictions are absent from ground truth."
        )

    # Keep only samples present in predictions
    df_y_true = df_y_true.loc[df_y_pred.index]

    # Remove ignored classes
    ignored_classes = [
        LCD.CLASSES[i] for i in LCD.IGNORED_CLASSES_IDX
    ]

    df_y_true = df_y_true.drop(
        columns=ignored_classes, errors="ignore"
    )
    df_y_pred = df_y_pred.drop(
        columns=ignored_classes, errors="ignore"
    )

    # Ensure identical column order
    df_y_pred = df_y_pred.loc[:, df_y_true.columns]

    if df_y_true.shape != df_y_pred.shape:
        raise ValueError("Shapes of ground truth and predictions differ.")

    score = epsilon_kl_divergence(
        df_y_true.to_numpy(),
        df_y_pred.to_numpy()
    )


    

    print(f"KL divergence: {score}")

    df_score = pd.DataFrame([{
        "gt_file": str(gt_file),
        "pred_file": str(pred_file),
        "score": score
    }])

    df_score.to_csv(out_csv, index=False)

    # --------------------------------------------------
    # 2. Pixel-wise confusion matrix
    # --------------------------------------------------
    # images of the set: IDs listed in gt_images, masks read in gt_mask_dir
    image_ids = read_image_ids(gt_images)
    print(f"\n{len(image_ids)} images listed in {gt_images.name}")

    # both parts of the evaluation should cover the same images
    pred_ids = set(df_y_pred.index.astype(np.int64))
    if pred_ids != set(image_ids):
        print(
            f"Warning: {pred_file.name} has {len(pred_ids)} samples, {gt_images.name} lists "
            f"{len(image_ids)} ({len(pred_ids - set(image_ids))} only in the predictions, "
            f"{len(set(image_ids) - pred_ids)} only in the image list)"
        )

    df_cm = build_confusion_matrix(
        gt_mask_dir=gt_mask_dir, pred_mask_dir=pred_mask_dir, image_ids=image_ids
    )


    metric_cols = ["Recall (%)", "Precision (%)", "F1-score (%)", "IoU (%)"]
    df_mean = df_cm[metric_cols].mean().rename("mean").rename_axis("metric")
    df_mean.to_csv(xp_dir / f"mean_metrics_{tag}_{set_name}.csv")
    print(df_mean)
 

    print("\nPixel-wise confusion matrix:")
    print(df_cm)
    print(f"\nPixel-wise results saved to: {confusion_out}")
    df_cm.to_csv(confusion_out)

    print(f"\nKL results saved to: {out_csv}")