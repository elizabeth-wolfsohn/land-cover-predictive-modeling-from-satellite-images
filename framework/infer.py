"""
Predict the targets using a trained model.
"""
import sys
from pathlib import Path
# make the "framework" package importable whatever the way the script is launched
# (VS Code "Run Python File", `python framework/train.py`, or `python -m framework.train`)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import argparse
import yaml
import random
from tifffile import TiffFile
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

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


def predict_as_vectors(model, dataset, steps=None, aggregation='argmax'):
    """Perform a forward pass over the dataset and bincount the prediction masks to return class vectors.
    Args:
        model (tf.keras.Model): model
        dataset (tf.data.Dataset): dataset to perform inference on
        steps (int, optional): the total number of steps (batches) in the dataset, used for the progress bar
        aggregation (str): 'argmax' counts the most confident class of every pixel (benchmark);
            'soft' averages the predicted probabilities over the pixels
    Returns:
        (pandas.DataFrame): predicted class distribution vectors for the dataset
    """
    def bincount_along_axis(arr, minlength=None, axis=-1):
        """Bincounts a tensor along an axis"""
        if minlength is None:
            minlength = tf.reduce_max(arr) + 1
        mask = tf.equal(arr[..., None], tf.range(minlength, dtype=arr.dtype))
        return tf.math.count_nonzero(mask, axis=axis-1 if axis < 0 else axis)

    predictions = []
    for batch in tqdm(dataset, total=steps):
        # predict a raster for each sample in the batch
        pred_raster = model.predict_on_batch(batch)
        if aggregation == 'soft':
            predictions.append(tf.reduce_mean(tf.cast(pred_raster, tf.float64), axis=(1, 2)))
            continue

        (batch_size, _, _, num_classes) = tuple(pred_raster.shape)
        pred_mask = tf.argmax(pred_raster, -1) # (bs, 256, 256)
        # bincount for each sample
        counts = bincount_along_axis(
            tf.reshape(pred_mask, (batch_size, -1)), minlength=num_classes, axis=-1
        )
        predictions.append(counts / tf.math.reduce_sum(counts, -1, keepdims=True))

    predictions = tf.concat(predictions, 0)
    return predictions.numpy()


def _parse_args():
    parser = argparse.ArgumentParser('Inference script')
    parser.add_argument('--config', '-c', type=str, default=str(ROOT_DIR/'infer_config.yaml'),
                        help="The YAML config file (default: infer_config.yaml at the repo root)")

    cli_args = parser.parse_args()
    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    config.dataset_folder = (ROOT_DIR/Path(config.dataset_folder).expanduser()).resolve()
    assert config.dataset_folder.is_dir(), f"dataset_folder not found: {config.dataset_folder}"
    config.xp_rootdir = (ROOT_DIR/Path(config.xp_rootdir).expanduser()).resolve()
    assert config.xp_rootdir.is_dir(), f"xp_rootdir not found: {config.xp_rootdir}"
    if config.xp_name == 'last':
        # get the most recently created experiment directory
        config.xp_dir = max((d for d in config.xp_rootdir.iterdir() if d.is_dir()),
                            key=lambda d: d.stat().st_mtime)
    else:
        config.xp_dir = config.xp_rootdir/config.xp_name
    assert config.xp_dir.is_dir()
    assert config.set in ('train', 'test', 'val')
    if not hasattr(config, 'aggregation'):
        config.aggregation = 'argmax'
    assert config.aggregation in ('argmax', 'soft')

    return config


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
    else:
        val_samples_s = pd.read_csv(config.xp_dir/'val_samples.csv').squeeze('columns')
        val_files = [config.dataset_folder/'train/images/{}.tif'.format(i) for i in val_samples_s]
        if config.set == 'train':
            test_files = [
                f for f in sorted(config.dataset_folder.glob('train/images/*.tif'))
                if f not in set(val_files)
            ]
        else:
            test_files = val_files
    testset_size = len(test_files)

    test_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, test_files)))
    # assert that samples are loaded in the right order (not shuffled)
    # for idx, (f, tensor) in enumerate(zip(test_files, test_dataset)):
    #     assert f == str(bytes.decode(tensor.numpy()))

    test_dataset = test_dataset.map(parse_image, num_parallel_calls=N_CPUS)\
        .map(load_image_test, num_parallel_calls=N_CPUS)\
        .repeat(1)\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Load the trained model saved to disk
    model = tf.keras.models.load_model(
        str(config.xp_dir/f'checkpoints/epoch{config.checkpoint_epoch}'), compile=False
    )

    print(f"Predict the vectors over the {config.set} dataset")
    y_pred = predict_as_vectors(model, test_dataset, steps=-(-testset_size // config.batch_size),
                                aggregation=config.aggregation)

    # get the samples ids
    ids_s = pd.Series([int(f.stem) for f in test_files], name='sample_id', dtype='uint32')
    df_y_pred = pd.DataFrame(
        y_pred, index=ids_s, columns=LCD.CLASSES
    )
    out_csv = config.xp_dir/f'epoch{config.checkpoint_epoch}_{config.set}_{config.aggregation}_predicted.csv'
    print(f"Saving prediction CSV to file {str(out_csv)}")
    df_y_pred.to_csv(out_csv, index=True, index_label='sample_id')
