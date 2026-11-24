"""
Build the group's local_dataset layout from the split CSVs, with symbolic links (no copy):
    local_dataset/{train,validation,test}/{images,masks}/<sample_id>.tif
        -> dataset_UNZIPPED/train/{images,masks}/<sample_id>.tif
Usage: python -m framework.make_local_dataset [--src ./dataset_UNZIPPED] [--dst ./local_dataset]
"""
from pathlib import Path
import argparse
import os
import pandas as pd

SPLITS = ('train', 'validation', 'test')

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Build local_dataset with symlinks')
    parser.add_argument('--src', type=Path, default=Path('./dataset_UNZIPPED'), help="original dataset folder")
    parser.add_argument('--dst', type=Path, default=Path('./local_dataset'), help="folder containing csv_files/")
    args = parser.parse_args()
    src, dst = args.src.expanduser().resolve(), args.dst.expanduser().resolve()

    for split in SPLITS:
        ids = pd.read_csv(dst/'csv_files'/f'{split}_images.csv')['sample_id'].astype(int)
        n_new = 0
        for sub in ('images', 'masks'):
            out_dir = dst/split/sub
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in ids:
                target, link = src/'train'/sub/f'{i}.tif', out_dir/f'{i}.tif'
                if not target.is_file():
                    raise FileNotFoundError(target)
                # relative links: the project folder can be moved
                rel = os.path.relpath(target, out_dir)
                if link.is_symlink() and os.readlink(link) == rel:
                    continue
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(rel)
                n_new += 1
            n_files = len(list(out_dir.glob('*.tif')))
            if n_files != len(ids):
                raise RuntimeError(f"{out_dir}: {n_files} .tif files, {len(ids)} expected (stray files?)")
        print(f"{split}: {len(ids)} images and masks ({n_new} links created)")
