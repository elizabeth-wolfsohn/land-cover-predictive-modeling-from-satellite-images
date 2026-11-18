from pathlib import Path
import argparse
import shutil

import pandas as pd


# Chemins connus
DEST_ROOT = Path("local_dataset")
SRC_ROOT = Path("dataset")

CSV_ROOT = DEST_ROOT / "csv_files"


def main(csv_name, set_type, column="sample_id"):

    csv_path = CSV_ROOT / csv_name

    # Sources
    src_images = SRC_ROOT / "images"
    src_masks = SRC_ROOT / "masks"

    # Destination
    dst_dataset = DEST_ROOT / set_type
    dst_images = dst_dataset / "images"
    dst_masks = dst_dataset / "masks"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    ids = pd.read_csv(csv_path)[column].astype(int).tolist()

    print(f"{len(ids)} IDs dans {csv_name}")

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    missing_images = []
    missing_masks = []

    for i in ids:
        img = src_images / f"{i}.tif"
        mask = src_masks / f"{i}.tif"

        if img.exists():
            shutil.copy2(img, dst_images / img.name)
        else:
            missing_images.append(i)

        if mask.exists():
            shutil.copy2(mask, dst_masks / mask.name)
        else:
            missing_masks.append(i)

    print(f"\nDataset créé : {dst_dataset}")
    print(f"Images copiées : {len(ids) - len(missing_images)}")
    print(f"Masques copiés : {len(ids) - len(missing_masks)}")

    if missing_images:
        print(f"Images introuvables : {missing_images}")

    if missing_masks:
        print(f"Masques introuvables : {missing_masks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True,
                        help="Nom du fichier CSV")

    parser.add_argument("--set-type", required=True,
                        choices=["train", "test", "validation"],
                        help="Type de dataset à créer")

    parser.add_argument("--column", default="sample_id")

    args = parser.parse_args()

    main(args.csv, args.set_type, args.column)
