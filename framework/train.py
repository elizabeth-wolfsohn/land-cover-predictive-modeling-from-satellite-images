"""
Train the model.
"""
from pathlib import Path
import datetime
import argparse
import shutil
import itertools
import yaml
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from framework.dataset import LandCoverData as LCD
from framework.dataset import LandCoverDataset, class_distribution, predict_distribution, input_normalization
from framework.eval import epsilon_kl_divergence
from framework.losses import build_loss
from framework.model import build_model
from framework.utils import YamlNamespace


def _optional_config_value(config, key, default=None):
    """Renvoie config.<key>, ou `default` si la clé est absente du fichier YAML."""
    try:
        value = getattr(config, key)
    except (AttributeError, KeyError):
        return default
    return default if value is None else value


def _parse_args():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--config', '-c', type=str, required=True, help="The YAML config file")
    cli_args = parser.parse_args()
    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    config.config_file = cli_args.config
    config.xp_rootdir = Path(config.xp_rootdir).expanduser()
    assert config.xp_rootdir.is_dir()
    config.dataset_folder = Path(config.dataset_folder).expanduser()
    assert config.dataset_folder.is_dir()
    for key in ('train_image_path', 'val_image_path', 'test_image_path'):
        setattr(config, key, Path(getattr(config, key)).expanduser())
        assert getattr(config, key).is_file(), getattr(config, key)
    assert config.model in ('unet', 'swin')
    assert config.pred_mode in ('argmax', 'soft')
    return config


def read_split(config):
    """Train / val / test image files from the group's split CSVs (column 'sample_id')."""
    splits = {name: pd.read_csv(getattr(config, f'{name}_image_path'))['sample_id'].astype(int).tolist()
              for name in ('train', 'val', 'test')}
    for name, ids in splits.items():
        assert len(set(ids)) == len(ids), f"{name}: duplicated sample ids"
    for a, b in itertools.combinations(splits, 2):
        overlap = set(splits[a]) & set(splits[b])
        assert not overlap, f"{len(overlap)} images both in {a} and {b}, e.g. {sorted(overlap)[:5]}"
    return {name: [config.dataset_folder/'train/images'/f'{i}.tif' for i in ids] for name, ids in splits.items()}


def build_class_weights(mode='uniform', power=0.5, max_weight=None, weights=None):
    """Loss class weights (as in the group's code): w(c) = freq(c)^-p, normalized to an average
    pixel weight of 1, clipped to max_weight. mode: 'uniform' (p=0), 'benchmark' (p=1, no clipping),
    'power' (p=power) or 'manual' (the `weights` list, one per class, given as is).
    "no_data" and "clouds" always get 0."""
    if mode == 'manual':
        assert weights is not None, "class_weighting.weights is required with mode 'manual'"
        weights = np.asarray(weights, dtype='float32')
        assert weights.shape == (LCD.N_CLASSES,), f"expected {LCD.N_CLASSES} class weights, got {weights.shape}"
        assert (weights[LCD.IGNORED_CLASSES_IDX] == 0).all(), "ignored classes must get a weight of 0"
        return weights
    p = {'uniform': 0., 'benchmark': 1., 'power': float(power)}[mode]
    if mode != 'power':
        max_weight = None
    weights = np.zeros(LCD.N_CLASSES)
    valid = [i for i in range(LCD.N_CLASSES) if i not in LCD.IGNORED_CLASSES_IDX]
    freq = LCD.TRAIN_CLASS_COUNTS[valid] / LCD.TRAIN_CLASS_COUNTS[valid].sum()
    w = np.where(freq > 0, freq, np.inf) ** -p  # absent classes get 0
    w /= (w * freq).sum()
    weights[valid] = w if max_weight is None else np.minimum(w, max_weight)
    return weights.astype('float32')


def kl_score(y_true, y_pred):
    """Challenge metric on (N, N_CLASSES) vectors: ignored classes dropped, then renormalized."""
    valid = [i for i in range(LCD.N_CLASSES) if i not in LCD.IGNORED_CLASSES_IDX]
    return epsilon_kl_divergence(y_true[:, valid], y_pred[:, valid])


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, autocast):
    """Return the validation metrics over a dataset: loss, KL scores (argmax and soft),
    pixel accuracy and mean IoU (ignored classes excluded)."""
    model.eval()
    total_loss, y_true, y_argmax, y_soft = 0., [], [], []
    conf = torch.zeros(LCD.N_CLASSES ** 2, dtype=torch.long, device=device)  # confusion matrix (true, pred)
    ignored = torch.tensor(LCD.IGNORED_CLASSES_IDX, device=device)
    for image, mask in tqdm(loader, desc='val', leave=False):
        image, mask = image.to(device), mask.to(device)
        mask = mask.squeeze(-1).long()
        with autocast():
            logits = model(image)
        total_loss += loss_fn(logits.float(), mask).item() * len(image)
        y_true.append(class_distribution(mask).cpu())
        y_argmax.append(predict_distribution(logits, 'argmax').cpu())
        y_soft.append(predict_distribution(logits, 'soft').cpu())
        pred = logits.float().index_fill(1, ignored, -torch.inf).argmax(1)
        valid = ~torch.isin(mask, ignored)
        conf += torch.bincount(mask[valid] * LCD.N_CLASSES + pred[valid], minlength=LCD.N_CLASSES ** 2)
    y_true = torch.cat(y_true).numpy()
    conf = conf.reshape(LCD.N_CLASSES, LCD.N_CLASSES).double().cpu()
    tp, union = conf.diag(), conf.sum(0) + conf.sum(1) - conf.diag()
    return dict(val_loss=total_loss / len(loader.dataset),
                val_kl_argmax=kl_score(y_true, torch.cat(y_argmax).numpy()),
                val_kl_soft=kl_score(y_true, torch.cat(y_soft).numpy()),
                val_acc=(tp.sum() / conf.sum()).item(),
                # mean over the classes present in the targets or the predictions
                val_miou=(tp[union > 0] / union[union > 0]).mean().item())


def mask_to_rgb(mask):
    lut = np.stack([LCD.CLASSES_COLORPALETTE[c] for c in range(LCD.N_CLASSES)]).astype('uint8')
    return lut[mask]


@torch.no_grad()
def save_plot(model, images, masks, filepath, device, autocast, norm):
    """Save a figure with the input image, the true mask and the predicted mask for a few samples."""
    model.eval()
    with autocast():
        logits = model(images.to(device)).float()
    logits[:, LCD.IGNORED_CLASSES_IDX] = -torch.inf
    pred = logits.argmax(1).cpu().numpy()
    fig, axs = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)), squeeze=False)
    for i in range(len(images)):
        rgb = images[i, :3].numpy().transpose(1, 2, 0) * norm[1][:3] + norm[0][:3]
        rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
        for ax, img, title in zip(axs[i], (rgb, mask_to_rgb(masks[i].numpy()), mask_to_rgb(pred[i])),
                                  ('Input Image', 'True Mask', 'Predicted Mask')):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':

    config = _parse_args()
    print(f'Config:\n{config}')
    # set random seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True
    # bfloat16 mixed precision on GPU, full precision on CPU
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')

    print('Instanciate train, validation and test datasets')
    split = read_split(config)
    if config.debug_samples:
        # small subset for quick tests (e.g. on CPU)
        n_eval = max(config.debug_samples // 4, config.batch_size)
        split = dict(train=split['train'][:config.debug_samples], val=split['val'][:n_eval], test=split['test'][:n_eval])
    train_files, val_files = split['train'], split['val']
    print(f"{len(train_files)} train samples, {len(val_files)} val samples, {len(split['test'])} test samples")

    norm = input_normalization(config)
    train_loader = DataLoader(LandCoverDataset(train_files, augment=True, norm=norm), batch_size=config.batch_size,
                              shuffle=True, drop_last=True, num_workers=config.num_workers,
                              pin_memory=True, persistent_workers=config.num_workers > 0)
    val_loader = DataLoader(LandCoverDataset(val_files, norm=norm), batch_size=config.batch_size,
                            num_workers=config.num_workers, pin_memory=True,
                            persistent_workers=config.num_workers > 0)

    # where to write files for this experiments
    xp_dir = config.xp_rootdir / datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    (xp_dir/'plots').mkdir(parents=True)
    (xp_dir/'checkpoints').mkdir()
    shutil.copy(config.config_file, xp_dir/'config.yaml')
    # save the samples of each set, read back by infer.py
    for name, files in split.items():
        pd.Series([int(f.stem) for f in files], name='sample_id', dtype='uint32')\
            .to_csv(xp_dir/f'{name}_samples.csv', index=False)
    print(f"Experiment directory: {xp_dir}")

    # keep a few validation samples for visualization
    plot_images, plot_masks = next(iter(DataLoader(val_loader.dataset, batch_size=5)))
    plot_masks = plot_masks.squeeze(-1)

    model = build_model(config).to(device)
    print(f"Model {config.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    cw = config.class_weighting
    class_weight = build_class_weights(cw.mode, cw.power, cw.max_weight,
                                       weights=_optional_config_value(cw, 'weights'))
    print(f"Will use class weights: {class_weight}")
    loss_cfg = _optional_config_value(config, 'loss')
    loss_fn = build_loss(
        loss_type=_optional_config_value(loss_cfg, 'type', 'ce'),
        n_classes=LCD.N_CLASSES,
        ignored_idx=LCD.IGNORED_CLASSES_IDX,
        ce_weight=_optional_config_value(loss_cfg, 'ce_weight', 1.0),
        kl_weight=_optional_config_value(loss_cfg, 'kl_weight', 1.0),
        class_weight=torch.from_numpy(class_weight),
    ).to(device)
    print(f"Loss: {loss_fn}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # linear warmup then cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, pct_start=0.05,
                                                    total_steps=config.epochs * len(train_loader))

    logs, best_kl = [], float('inf')
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, train_correct, train_valid = 0., 0, 0
        ignored = torch.tensor(LCD.IGNORED_CLASSES_IDX, device=device)
        for image, mask in tqdm(train_loader, desc=f'epoch {epoch}/{config.epochs}'):
            image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            mask = mask.squeeze(-1).long()
            with autocast():
                logits = model(image)
            loss = loss_fn(logits.float(), mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            # pixel accuracy on the training batches (augmented, model in train mode)
            pred = logits.detach().float().index_fill(1, ignored, -torch.inf).argmax(1)
            valid = ~torch.isin(mask, ignored)
            train_correct += (pred[valid] == mask[valid]).sum().item()
            train_valid += valid.sum().item()
        train_loss /= len(train_loader)

        val = evaluate(model, val_loader, loss_fn, device, autocast)
        logs.append(dict(epoch=epoch, lr=scheduler.get_last_lr()[0], loss=train_loss,
                         acc=train_correct / max(train_valid, 1), **val))
        pd.DataFrame(logs).to_csv(xp_dir/'fit_logs.csv', index=False)
        print(f"epoch {epoch}: " + " ".join(f"{k}={v:.5f}" for k, v in logs[-1].items() if k not in ('epoch', 'lr')))

        torch.save(model.state_dict(), xp_dir/'checkpoints/last.pt')
        val_kl = val[f'val_kl_{config.pred_mode}']
        if val_kl < best_kl:
            best_kl = val_kl
            torch.save(model.state_dict(), xp_dir/'checkpoints/best.pt')
            print(f"New best val_kl_{config.pred_mode}={val_kl:.5f}, saved checkpoints/best.pt")
        save_plot(model, plot_images, plot_masks, xp_dir/f'plots/epoch{epoch}.png', device, autocast, norm)

    if config.evaluate_on_test:
        # internal test set, never used to choose the model: evaluated once with best.pt
        model.load_state_dict(torch.load(xp_dir/'checkpoints/best.pt', map_location=device))
        test_loader = DataLoader(LandCoverDataset(split['test'], norm=norm), batch_size=config.batch_size,
                                 num_workers=config.num_workers, pin_memory=True)
        test = {k.replace('val_', 'test_'): v for k, v in evaluate(model, test_loader, loss_fn, device, autocast).items()}
        pd.DataFrame([test]).to_csv(xp_dir/'test_metrics.csv', index=False)
        print("Test (best.pt): " + " ".join(f"{k}={v:.5f}" for k, v in test.items()))
