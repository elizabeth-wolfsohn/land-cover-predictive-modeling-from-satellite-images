"""
Predict the targets using a trained model.
"""
from pathlib import Path
import argparse
import math
import yaml
import random
from tifffile import TiffFile, imwrite
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import time
import json

from framework.dataset import LandCoverData as LCD
from framework.utils import YamlNamespace


def numpy_parse_image(image_path):
    """Load an image as numpy array
    Args:
        image_path (bytes): path to image
    Returns:
        numpy.array[uint8]: the image array
    """
    image_path = Path(bytes.decode(image_path))
    with TiffFile(image_path) as tifi:
        image = tifi.asarray()
    return image

@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def parse_image(image_path):
    """Wraps the parse_image function as a TF function"""
    image, = tf.numpy_function(numpy_parse_image, (image_path,), (tf.uint16,))
    image.set_shape([LCD.IMG_SIZE, LCD.IMG_SIZE, LCD.N_CHANNELS])
    return image

@tf.function
def normalize(input_image):
    """Rescale the pixel values of the images between 0.0 and 1.0"""
    image = tf.cast(input_image, tf.float32) / LCD.TRAIN_PIXELS_MAX
    return image

@tf.function
def load_image_test(input_image):
    """Normalize test image"""
    image = normalize(input_image)
    return image


def predict_as_vectors(model, dataset, files, pred_mask_dir, steps=None):
    """Forward pass over the dataset: saves the predicted masks and returns class proportions.
    Args:
        model (tf.keras.Model): model
        dataset (tf.data.Dataset): dataset to perform inference on (not shuffled)
        files (list[Path]): image files, in the same order as the dataset
        pred_mask_dir (Path): directory where predicted masks are saved as <stem>.tif
        steps (int, optional): total number of batches, used for the progress bar
    Returns:
        (numpy.ndarray): predicted class distribution vectors for the dataset
    """
    def bincount_along_axis(arr, minlength=None, axis=-1):
        """Bincounts a tensor along an axis"""
        if minlength is None:
            minlength = tf.reduce_max(arr) + 1
        mask = tf.equal(arr[..., None], tf.range(minlength, dtype=arr.dtype))
        return tf.math.count_nonzero(mask, axis=axis-1 if axis < 0 else axis)

    print(f"Saving prediction masks to directory {pred_mask_dir}")

    predictions = []
    batch_times = []
    idx = 0  # position in the list of files
    for batch in tqdm(dataset, total=steps):
        # predict a raster for each sample in the batch
        t0 = time.perf_counter()
        pred_raster = model.predict_on_batch(batch)
        dt = time.perf_counter() - t0

        (batch_size, _, _, num_classes) = tuple(pred_raster.shape)
        batch_times.append((dt, batch_size))
        pred_mask = tf.argmax(pred_raster, -1)  # (bs, 256, 256)
        

        # save the predicted masks, named like the input images
        masks_np = pred_mask.numpy().astype(np.uint8)
        for m, f in zip(masks_np, files[idx:idx + batch_size]):
            imwrite(pred_mask_dir / f"{f.stem}.tif", m)
        idx += batch_size

        # bincount for each sample
        counts = bincount_along_axis(
            tf.reshape(pred_mask, (batch_size, -1)), minlength=num_classes, axis=-1
        )
        predictions.append(counts / tf.math.reduce_sum(counts, -1, keepdims=True))

    predictions = tf.concat(predictions, 0)
    return predictions.numpy(), batch_times


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
    assert config.set in ('train', 'test', 'val')

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

    import multiprocessing

    config = _parse_args()
    print(f'Config:\n{config}')
    # set random seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    N_CPUS = multiprocessing.cpu_count()

    print(f"Instanciate {config.set} dataset")

    if config.set == 'test':
        test_files = sorted(config.dataset_folder.glob('test/images/*.tif'))
    elif config.set == 'val':
        # same folder as the one used during training
        test_files = sorted((config.dataset_folder / 'validation/images').glob('*.tif'))
    else:  # 'train'
        test_files = sorted(config.dataset_folder.glob('train/images/*.tif'))

    testset_size = len(test_files)

    test_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, test_files)))

    test_dataset = test_dataset.map(parse_image, num_parallel_calls=N_CPUS)\
        .map(load_image_test, num_parallel_calls=N_CPUS)\
        .repeat(1)\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # checkpoint: 'best' or an epoch number
    if str(config.checkpoint_epoch) == 'best':
        tag = 'best'
        checkpoint_path = config.xp_dir / 'checkpoints' / 'best_model.keras'
    else:
        tag = f'epoch{int(config.checkpoint_epoch)}'
        checkpoint_path = config.xp_dir / 'checkpoints' / f'epoch{int(config.checkpoint_epoch):02d}.keras'

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = tf.keras.models.load_model(str(checkpoint_path), compile=False)
    print(f"Successfully loaded model: {checkpoint_path}")

    pred_mask_dir = config.xp_dir / f'{tag}_{config.set}_pred_masks'
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Predict the vectors over the {config.set} dataset")
    y_pred, batch_times= predict_as_vectors(
        model, test_dataset,
        files=test_files,
        pred_mask_dir=pred_mask_dir,
        steps=math.ceil(testset_size / config.batch_size),
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
    out_csv = config.xp_dir / f'{tag}_{config.set}_predicted.csv'
    print(f"Saving prediction CSV to file {out_csv}")
    df_y_pred.to_csv(out_csv, index=True, index_label='sample_id')