"""
Predict the targets using a trained model.
(PyTorch port of infer_tf.py, which is the reference)
"""
from pathlib import Path
import argparse
import yaml
import random
from tifffile import imwrite
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import time
import json
 
from framework.dataset import LandCoverData as LCD
from framework.dataset import LandCoverDataset, input_normalization
from framework.model import build_model
from framework.utils import YamlNamespace
 
 
@torch.no_grad()
def predict_as_vectors(model, loader, files, pred_mask_dir, device):
    """Forward pass over the dataset: saves the predicted masks and returns class proportions.
    Args:
        model (torch.nn.Module): model, in eval mode
        loader (DataLoader): dataset to perform inference on (not shuffled)
        files (list[Path]): image files, in the same order as the dataset
        pred_mask_dir (Path or None): directory where predicted masks are saved as <stem>.tif
            (None: masks not saved, for the official test set)
        device (torch.device): device of the model
    Returns:
        (numpy.ndarray): predicted class distribution vectors for the dataset
        (list[tuple]): (duration in s, batch size) of each batch
    """
    if pred_mask_dir is not None:
        print(f"Saving prediction masks to directory {pred_mask_dir}")
 
    predictions = []
    batch_times = []
    idx = 0  # position in the list of files
    for image in tqdm(loader):
        # predict a raster for each sample in the batch
        # timed like model.predict_on_batch: host->device copy, forward pass, result back on the host
        # ponytail: only the argmax mask comes back to the host, not the (bs, 10, 256, 256) raster
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            pred_raster = model(image.to(device))
        pred_mask = pred_raster.float().argmax(1).cpu()  # (bs, 256, 256), argmax over the 10 classes
        dt = time.perf_counter() - t0  # .cpu() waits for the GPU
 
        (batch_size, num_classes, _, _) = tuple(pred_raster.shape)
        batch_times.append((dt, batch_size))
 
        # save the predicted masks, named like the input images
        masks_np = pred_mask.numpy().astype(np.uint8)
        if pred_mask_dir is not None:
            for m, f in zip(masks_np, files[idx:idx + batch_size]):
                imwrite(pred_mask_dir / f"{f.stem}.tif", m)
        idx += batch_size
 
        # bincount for each sample
        counts = np.stack([np.bincount(m.ravel(), minlength=num_classes) for m in masks_np])
        predictions.append(counts / counts.sum(-1, keepdims=True))
 
    predictions = np.concatenate(predictions, 0)
    return predictions, batch_times
 
 
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
        # most recently modified xp directory
        config.xp_dir = max(
            (d for d in config.xp_rootdir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime
        )
    else:
        config.xp_dir = config.xp_rootdir / config.xp_name
    assert config.xp_dir.is_dir()
    # 'final_test': official test set of the challenge (no masks), the CSV to submit
    assert config.set in ('train', 'test', 'val', 'final_test')
    if config.set == 'final_test':
        config.final_test_folder = Path(config.final_test_folder).expanduser()
        assert (config.final_test_folder/'images').is_dir(), config.final_test_folder/'images'
    # only the best checkpoint is saved by train.py
    assert str(config.checkpoint_epoch) == 'best', "checkpoint_epoch: only 'best' is supported"
 
    return config
 
 
def summarize_timings(batch_times, warmup=1):
    """Statistiques d'inférence. Le(s) premier(s) batch(s) sont ignorés
    (tracing du graphe / initialisation GPU)."""
    bt = batch_times[warmup:] if len(batch_times) > warmup else batch_times
    durations = np.array([d for d, _ in bt])
    n_images = sum(n for _, n in bt)
    per_image_ms = np.array([d / n * 1000 for d, n in bt])
    return {
        "n_batches_measured": len(bt),
        "n_images_measured": int(n_images),
        "first_batch_s (warmup)": float(batch_times[0][0]),
        "total_inference_s": float(durations.sum()),
        "throughput_img_per_s": float(n_images / durations.sum()),
        "latency_ms_per_image_mean": float(per_image_ms.mean()),
        "latency_ms_per_image_median": float(np.median(per_image_ms)),
        "latency_ms_per_image_p95": float(np.percentile(per_image_ms, 95)),
        "batch_size": int(bt[0][1]),
    }
 
 
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
 
    if config.set == 'test':
        test_files = sorted(config.dataset_folder.glob('test/images/*.tif'))
    elif config.set == 'val':
        # same folder as the one used during training
        test_files = sorted((config.dataset_folder / 'validation/images').glob('*.tif'))
    elif config.set == 'final_test':
        # official test set of the challenge, images only (dataset_UNZIPPED/test/images)
        test_files = sorted(config.final_test_folder.glob('images/*.tif'))
    else:  # 'train'
        test_files = sorted(config.dataset_folder.glob('train/images/*.tif'))
    print(f"{len(test_files)} images")
 
    # images only, normalized as in load_image_test (image / TRAIN_PIXELS_MAX)
    with open(config.xp_dir/'config.yaml') as f:
        train_config = YamlNamespace(yaml.load(f, Loader=yaml.FullLoader))
    test_dataset = LandCoverDataset(test_files, with_mask=False, norm=input_normalization(train_config))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True)
 
    # checkpoint: 'best'
    tag = 'best'
    checkpoint_path = config.xp_dir / 'checkpoints' / 'best.pt'
 
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
 
    # rebuild the model from the training config, then load the trained weights
    model = build_model(train_config, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Successfully loaded model: {checkpoint_path}")
 
    if config.set == 'final_test':
        pred_mask_dir = None  # no ground truth: only the submission CSV is needed
    else:
        pred_mask_dir = config.xp_dir / f'{tag}_{config.set}_pred_masks'
        pred_mask_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"Predict the vectors over the {config.set} dataset")
    y_pred, batch_times = predict_as_vectors(
        model, test_loader,
        files=test_files,
        pred_mask_dir=pred_mask_dir,
        device=device,
    )
 
    stats = summarize_timings(batch_times, warmup=1)
    print("Inference timing:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    with open(config.xp_dir / f'{tag}_{config.set}_inference_timing.json', 'w') as f:
        json.dump(stats, f, indent=2)
 
    # get the samples ids
    ids_s = pd.Series([int(f.stem) for f in test_files], name='sample_id', dtype='uint32')
    df_y_pred = pd.DataFrame(y_pred, index=ids_s, columns=LCD.CLASSES)
    if config.set == 'final_test':
        # submission: rows in increasing sample_id order (sorted() above is in text order: 10 < 2)
        df_y_pred = df_y_pred.sort_index()
    out_csv = config.xp_dir / f'{tag}_{config.set}_predicted.csv'
    print(f"Saving prediction CSV to file {out_csv}")
    df_y_pred.to_csv(out_csv, index=True, index_label='sample_id')
 