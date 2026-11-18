"""
Train the model.

The optional preprocessing (tiling + re-sampling, augmentation, spectral indices) lives in
framework/preprocessing/ and is enabled from the YAML config. Without the corresponding
sections, the training is the benchmark one.
"""
from pathlib import Path
import datetime
import argparse
import shutil
import yaml
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from framework.dataset import LandCoverData as LCD
from framework.dataset import parse_image, load_image_train, load_image_test
from framework.model import UNet
from framework.tensorflow_utils import plot_predictions
from framework.utils import YamlNamespace
from framework.preprocessing.config import cfg_get, section_enabled
from framework.preprocessing.class_weights import compute_class_weights, make_sample_weight_fn

# config sections that switch to the preprocessing pipeline (framework/preprocessing/pipeline.py)
PREPROCESSING_SECTIONS = ('tiling', 'augmentation', 'spectral_indices')


class PlotCallback(tf.keras.callbacks.Callback):
    """A callback used to display sample predictions during training."""
    from IPython.display import clear_output

    def __init__(self, dataset: tf.data.Dataset=None,
                 sample_batch: tf.Tensor=None,
                 save_folder: Path=None,
                 num: int=1,
                 ipython_mode: bool=False):
        super(PlotCallback, self).__init__()
        self.dataset = dataset
        self.sample_batch = sample_batch
        self.save_folder = save_folder
        self.num = num
        self.ipython_mode = ipython_mode

    def on_epoch_begin(self, epoch, logs=None):
        if self.ipython_mode:
            self.clear_output(wait=True)
        if self.save_folder:
            save_filepaths = [self.save_folder/f'plot_{n}_epoch{epoch}.png' for n in range(1, self.num+1)]
        else:
            save_filepaths = None
        plot_predictions(self.model, self.dataset, self.sample_batch, num=self.num, save_filepaths=save_filepaths)


def _parse_args():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--config', '-c', type=str, required=True, help="The YAML config file")
    cli_args = parser.parse_args()
    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    config.config_path = Path(cli_args.config)
    config.xp_rootdir = Path(config.xp_rootdir).expanduser()
    assert config.xp_rootdir.is_dir(), f"xp_rootdir {config.xp_rootdir} does not exist"
    config.dataset_folder = Path(config.dataset_folder).expanduser()
    assert config.dataset_folder.is_dir(), f"dataset_folder {config.dataset_folder} does not exist"
    if config.val_samples_csv is not None:
        config.val_samples_csv = Path(config.val_samples_csv).expanduser()
        assert config.val_samples_csv.is_file(), f"val_samples_csv {config.val_samples_csv} does not exist"

    return config


def _legacy_datasets(config, train_files, val_files, n_cpus):
    """The benchmark tf.data pipeline (used when no preprocessing section is enabled)."""
    train_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, train_files)))\
        .map(parse_image, num_parallel_calls=n_cpus)\
        .map(load_image_train, num_parallel_calls=n_cpus)\
        .shuffle(buffer_size=1024, seed=config.seed)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, val_files)))\
        .map(parse_image, num_parallel_calls=n_cpus)\
        .map(load_image_test, num_parallel_calls=n_cpus)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    info = {
        'steps_per_epoch': len(train_files) // config.batch_size,
        'input_shape': (LCD.IMG_SIZE, LCD.IMG_SIZE, LCD.N_CHANNELS),
        'class_counts': None,
        'description': f"{len(train_files)} whole images (benchmark pipeline)",
    }
    return train_dataset, val_dataset, info


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

    print('Instanciate train and validation datasets')
    train_files = list(config.dataset_folder.glob('train/images/*.tif'))
    # shuffle list of training samples files
    train_files = random.sample(train_files, len(train_files))
    devset_size = len(train_files)
    # validation set
    if config.val_samples_csv is not None:
        # read the validation samples
        val_samples_s = pd.read_csv(config.val_samples_csv)['sample_id']
        val_files = [config.dataset_folder/'train/images/{}.tif'.format(i) for i in val_samples_s]
        val_set = set(val_files)
        train_files = [f for f in train_files if f not in val_set]
        valset_size = len(val_files)
        trainset_size = len(train_files)
        assert valset_size + trainset_size == devset_size
    else:
        # generate a hold-out validation set from the training set
        valset_size = int(len(train_files) * 0.1)
        train_files, val_files = train_files[valset_size:], train_files[:valset_size]
        trainset_size = len(train_files)

    # where to write files for this experiments
    xp_dir = config.xp_rootdir / datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    (xp_dir/'tensorboard').mkdir(parents=True)
    (xp_dir/'plots').mkdir()
    (xp_dir/'checkpoints').mkdir()
    # save the config (infer.py reads it back, e.g. to compute the same spectral indices)
    shutil.copy(config.config_path, xp_dir/'train_config.yaml')
    # save the validation samples to a CSV
    val_samples_s = pd.Series([int(f.stem) for f in val_files], name='sample_id', dtype='uint32')
    val_samples_s.to_csv(xp_dir/'val_samples.csv', index=False)

    use_preprocessing = any(section_enabled(config, s) for s in PREPROCESSING_SECTIONS)
    if use_preprocessing:
        from framework.preprocessing.pipeline import build_datasets
        train_dataset, val_dataset, data_info = build_datasets(
            config, train_files, val_files, xp_dir, n_cpus=N_CPUS)
    else:
        train_dataset, val_dataset, data_info = _legacy_datasets(config, train_files, val_files, N_CPUS)
    print(f"Training data: {data_info['description']}")

    # keep a validation minibatch for visualization (fixed, 256x256, same images every epoch)
    for image, mask in val_dataset.take(1):
        sample_batch = (image[:5, ...], mask[:5, ...])

    # compute class weights for the loss, applied as per-pixel sample weights
    # (Keras' fit(class_weight=...) does not support segmentation targets)
    # note: the weights of the classes "no_data"(0) and "clouds"(1) are set to 0 to ignore them
    counts_source = cfg_get(config, 'class_weighting.counts_source', 'manifest')
    if counts_source == 'manifest' and data_info['class_counts'] is not None:
        class_counts = data_info['class_counts']   # effective distribution after re-sampling
    else:
        class_counts = LCD.TRAIN_CLASS_COUNTS      # static training set distribution
    class_weight = compute_class_weights(
        class_counts,
        power=float(cfg_get(config, 'class_weighting.power', 0.5)),
        max_weight=float(cfg_get(config, 'class_weighting.max_weight', 10.0)),
    )
    print("Will use class weights:\n" + "\n".join(
        f"  {name:<12s} {w:.4f}" for name, w in zip(LCD.CLASSES, class_weight)))
    np.savetxt(xp_dir/'class_weights.txt', class_weight)
    train_dataset = train_dataset.map(make_sample_weight_fn(class_weight))

    callbacks = [
        PlotCallback(sample_batch=sample_batch, save_folder=xp_dir/'plots', num=5),
        tf.keras.callbacks.TensorBoard(
            log_dir=xp_dir/'tensorboard',
            update_freq='epoch'
        ),
        # tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(xp_dir/'checkpoints/epoch{epoch}'), save_best_only=False, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=(xp_dir/'fit_logs.csv')
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=20,
            factor=0.5,
            verbose=1,
        )
    ]
    # create the U-Net model to train
    unet_kwargs = dict(
        input_shape=data_info['input_shape'],
        num_classes=LCD.N_CLASSES,
        num_layers=2
    )
    print(f"Creating U-Net with arguments: {unet_kwargs}")
    model = UNet(**unet_kwargs)
    print(model.summary())

    # get optimizer, loss, and compile model for training
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    print("Compile model")
    model.compile(optimizer=optimizer,
                  loss=loss,
                  # unweighted cross-entropy: comparable between train ('ce') and validation ('val_ce')
                  metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(name="ce")])
                  # metrics = [tf.keras.metrics.Precision(),
                  #            tf.keras.metrics.Recall(),
                  #            tf.keras.metrics.MeanIoU(num_classes=LCD.N_CLASSES)]) # TODO segmentation metrics

    # Launch training
    print(f"Steps per epoch: {data_info['steps_per_epoch']}")
    model_history = model.fit(train_dataset, epochs=config.epochs,
                              callbacks=callbacks,
                              steps_per_epoch=data_info['steps_per_epoch'],
                              validation_data=val_dataset,
                              validation_steps=valset_size // config.batch_size
                              )
