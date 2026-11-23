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
from sklearn.metrics import confusion_matrix

from framework.dataset import LandCoverData as LCD


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


def build_confusion_matrix(gt_mask_dir, pred_mask_dir):
    """Build a pixel-wise confusion matrix with accuracy and F1-score."""

    class_ids = np.arange(LCD.N_CLASSES)

    gt_files = sorted(Path(gt_mask_dir).glob("*.tif"))

    if not gt_files:
        raise ValueError(f"No TIFF masks found in {gt_mask_dir}")

    all_y_true = []
    all_y_pred = []

    for gt_file in gt_files:
        pred_file = Path(pred_mask_dir) / gt_file.name

        if not pred_file.is_file():
            raise FileNotFoundError(
                f"Prediction mask not found: {pred_file}"
            )

        with TiffFile(gt_file) as t:
            y_true = t.asarray().ravel()

        with TiffFile(pred_file) as t:
            y_pred = t.asarray().ravel()

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Different mask dimensions for {gt_file.name}: "
                f"{y_true.shape} vs {y_pred.shape}"
            )

        # Ignore pixels belonging to ignored ground-truth classes
        valid = ~np.isin(y_true, LCD.IGNORED_CLASSES_IDX)

        all_y_true.append(y_true[valid])
        all_y_pred.append(y_pred[valid])

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    # Check class IDs
    if not np.all(np.isin(y_true, class_ids)):
        raise ValueError("Unknown class IDs in ground-truth masks.")

    if not np.all(np.isin(y_pred, class_ids)):
        raise ValueError("Unknown class IDs in predicted masks.")

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids
    )

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

    pred_file = xp_dir / prediction_file_name
    pred_mask_dir = xp_dir / f"{tag}_{set_name}_pred_masks"

    gt_mask_dir =dataset_folder/ dataset_set/"masks"

    #gt_mask_dir = dataset_folder / dataset_set / "masks"
    #pred_mask_dir = xp_dir / f"{set_name}_predicted_masks"

    out_csv = xp_dir / f"eval_{tag}_{set_name}_results.csv"
    confusion_out = xp_dir / f"confusion_{tag}_{set_name}.csv"

    # Check input files and directories
    if not gt_file.is_file():
        raise FileNotFoundError(gt_file)

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
    df_cm=build_confusion_matrix(gt_mask_dir=gt_mask_dir,  pred_mask_dir=pred_mask_dir)


    metric_cols = ["Recall (%)", "Precision (%)", "F1-score (%)", "IoU (%)"]
    df_mean = df_cm[metric_cols].mean().rename("mean").rename_axis("metric")
    df_mean.to_csv(xp_dir / f"mean_metrics_{tag}_{set_name}.csv")
    print(df_mean)
 

    print("\nPixel-wise confusion matrix:")
    print(df_cm)
    print(f"\nPixel-wise results saved to: {confusion_out}")
    df_cm.to_csv(confusion_out)

    print(f"\nKL results saved to: {out_csv}")

