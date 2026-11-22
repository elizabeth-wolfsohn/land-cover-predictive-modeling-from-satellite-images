"""
Predict the targets using a trained model.
"""
from pathlib import Path
import argparse
import time
import yaml
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from framework.dataset import LandCoverData as LCD
from framework.dataset import LandCoverDataset, class_distribution, predict_distribution, input_normalization
from framework.model import build_model
from framework.utils import YamlNamespace


# test-time augmentation: (number of 90 degrees rotations, horizontal flip) of each pass
TTA = {'none': [(0, False)], 'flip': [(0, False), (0, True)],
       'd4': [(k, f) for f in (False, True) for k in range(4)]}


def predict_tta(model, image, tta='none'):
    """Logits (B, C, H, W) averaged over the TTA transforms, mapped back to the original orientation."""
    # ponytail: averages logits (not probabilities), simpler and close in practice
    out = 0
    for k, flip in TTA[tta]:
        x = image.flip(-1) if flip else image
        y = model(x.rot90(k, (-2, -1))).float().rot90(-k, (-2, -1))
        out = out + (y.flip(-1) if flip else y)
    return out / len(TTA[tta])


def _parse_args():
    parser = argparse.ArgumentParser('Inference script')
    parser.add_argument('--config', '-c', type=str, required=True, help="The YAML config file")

    cli_args = parser.parse_args()
    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    config.dataset_folder = Path(config.dataset_folder).expanduser()
    assert config.dataset_folder.is_dir()
    config.xp_rootdir = Path(config.xp_rootdir).expanduser()
    assert config.xp_rootdir.is_dir()
    if config.xp_name == 'last':
        # get last created xp directory
        config.xp_dir = max((d for d in config.xp_rootdir.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    else:
        config.xp_dir = config.xp_rootdir/config.xp_name
    assert config.xp_dir.is_dir()
    assert config.set in ('train', 'val', 'test', 'final_test')
    assert config.tta in TTA
    assert config.checkpoint in ('best', 'last')
    assert config.pred_mode in ('argmax', 'soft')

    return config


if __name__ == '__main__':

    config = _parse_args()
    print(f'Config:\n{config}')
    # set random seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Instanciate {config.set} dataset")
    if config.set == 'final_test':
        test_files = sorted(config.dataset_folder.glob('test/images/*.tif'))
    else:
        ids = pd.read_csv(config.xp_dir/f'{config.set}_samples.csv')['sample_id']
        test_files = [config.dataset_folder/'train/images'/f'{i}.tif' for i in ids]
    # rebuild the model (and its input normalization) from the training config, load the trained weights
    with open(config.xp_dir/'config.yaml') as f:
        train_config = YamlNamespace(yaml.load(f, Loader=yaml.FullLoader))
    with_mask = config.set != 'final_test'
    loader = DataLoader(LandCoverDataset(test_files, with_mask=with_mask, norm=input_normalization(train_config)),
                        batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    model = build_model(train_config, pretrained=False).to(device).eval()
    checkpoint = config.xp_dir/f'checkpoints/{config.checkpoint}.pt'
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"Loaded {checkpoint}")

    print(f"Predict the vectors over the {config.set} dataset")
    y_pred, y_true = [], []
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for batch in tqdm(loader):
            image = (batch[0] if with_mask else batch).to(device)
            y_pred.append(predict_distribution(predict_tta(model, image, config.tta), config.pred_mode).cpu())
            if with_mask:
                y_true.append(class_distribution(batch[1]))
    # ponytail: end-to-end time (disk reading + first-batch warmup included); .cpu() above syncs the GPU
    dt = time.perf_counter() - t0
    print(f"Inference: {dt:.1f} s, {len(test_files) / dt:.1f} images/s, {1000 * dt / len(test_files):.2f} ms/image")

    # get the samples ids
    ids_s = pd.Series([int(f.stem) for f in test_files], name='sample_id', dtype='uint32')
    tta = '' if config.tta == 'none' else f'_tta-{config.tta}'
    out_csv = config.xp_dir/f'{config.checkpoint}_{config.pred_mode}{tta}_{config.set}_predicted.csv'
    print(f"Saving prediction CSV to file {str(out_csv)}")
    pd.DataFrame(torch.cat(y_pred).numpy(), index=ids_s, columns=LCD.CLASSES)\
        .to_csv(out_csv, index=True, index_label='sample_id')
    if with_mask:
        # ground-truth vectors computed from the masks, to use with eval.py
        gt_csv = config.xp_dir/f'{config.set}_true.csv'
        print(f"Saving ground-truth CSV to file {str(gt_csv)}")
        pd.DataFrame(torch.cat(y_true).numpy(), index=ids_s, columns=LCD.CLASSES)\
            .to_csv(gt_csv, index=True, index_label='sample_id')
