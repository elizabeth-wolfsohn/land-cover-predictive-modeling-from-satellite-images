"""
Train the model.
"""
import sys
from pathlib import Path
# make the "framework" package importable whatever the way the script is launched
# (VS Code "Run Python File", `python framework/train.py`, or `python -m framework.train`)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import datetime
import argparse
import yaml
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from framework.dataset import LandCoverData as LCD
from framework.dataset import parse_image, load_image_train, load_image_test
from framework.model import build_model
from framework.metrics import PixelAccuracy, ClassDistributionKL
from framework.visualization import (
    plot_dataset_samples, plot_predictions, plot_training_curves, open_file
)
from framework.utils import YamlNamespace

class VisualizationCallback(tf.keras.callbacks.Callback):
    """At the end of every epoch, saves:
    - the predictions of the model on a fixed set of validation images (maps + proportions)
    - the training curves (loss, pixel accuracy, KL divergence)
    """

    def __init__(self, images, masks, plots_dir: Path, every: int = 1):
        super().__init__()
        self.images, self.masks = images, masks
        self.plots_dir = plots_dir
        self.every = every
        self.history = []
        self.last_predictions_png = None
        self.curves_png = plots_dir/'training_curves.png'

    def on_epoch_end(self, epoch, logs=None):
        self.history.append({'epoch': epoch, **(logs or {})})
        try:
            plot_training_curves(self.history, self.curves_png)
            if (epoch + 1) % self.every == 0:
                probs = self.model.predict(self.images, verbose=0)
                self.last_predictions_png = plot_predictions(
                    self.images, self.masks, probs,
                    self.plots_dir/f'predictions_epoch{epoch + 1:03d}.png',
                    title=f'Images de validation - époque {epoch + 1}'
                )
        except Exception as e:  # a plotting problem must never stop the training
            print(f"[VisualizationCallback] plotting failed: {e}")


def _parse_args():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--config', '-c', type=str, default=str(ROOT_DIR/'train_config.yaml'),
                        help="The YAML config file (default: train_config.yaml at the repo root)")
    cli_args = parser.parse_args()
    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    # relative paths in the config are relative to the repo root
    config.xp_rootdir = (ROOT_DIR/Path(config.xp_rootdir).expanduser()).resolve()
    config.xp_rootdir.mkdir(parents=True, exist_ok=True)
    config.dataset_folder = (ROOT_DIR/Path(config.dataset_folder).expanduser()).resolve()
    if not config.dataset_folder.is_dir():
        raise FileNotFoundError(f"dataset_folder not found: {config.dataset_folder} "
                                "(set 'dataset_folder' in the YAML config)")
    if config.val_samples_csv is not None:
        config.val_samples_csv = (ROOT_DIR/Path(config.val_samples_csv).expanduser()).resolve()
        assert config.val_samples_csv.is_file()
    # defaults for options added after the original benchmark
    if not hasattr(config, 'model') or config.model is None:
        config.model = YamlNamespace({'name': 'unet', 'kwargs': {'num_layers': 2}})
    if not hasattr(config, 'class_weighting'):
        config.class_weighting = None
    if not hasattr(config, 'steps_per_epoch'):
        config.steps_per_epoch = None
    if not hasattr(config, 'lr_patience'):
        config.lr_patience = 20
    if not hasattr(config, 'max_train_samples'):
        config.max_train_samples = None
    if not hasattr(config, 'max_val_samples'):
        config.max_val_samples = None
    if not hasattr(config, 'show_plots'):
        config.show_plots = True
    if not hasattr(config, 'num_dataset_samples'):
        config.num_dataset_samples = 10
    if not hasattr(config, 'num_prediction_samples'):
        config.num_prediction_samples = 4

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

    print('Instanciate train and validation datasets')
    train_files = list(config.dataset_folder.glob('train/images/*.tif'))
    # shuffle list of training samples files
    train_files = random.sample(train_files, len(train_files))
    devset_size = len(train_files)
    # validation set
    if config.val_samples_csv is not None:
        # read the validation samples
        val_samples_s = pd.read_csv(config.val_samples_csv).squeeze('columns')
        val_files = [config.dataset_folder/'train/images/{}.tif'.format(i) for i in val_samples_s]
        train_files = [f for f in train_files if f not in set(val_files)]
        valset_size = len(val_files)
        trainset_size = len(train_files)
        assert valset_size + trainset_size == devset_size
    else:
        # generate a hold-out validation set from the training set
        valset_size = int(len(train_files) * 0.1)
        train_files, val_files = train_files[valset_size:], train_files[:valset_size]
        trainset_size = len(train_files)

    # optionally train / validate on a random subset of the images (faster experiments)
    # the files are already shuffled with the seed, so the subset is random and reproducible
    if config.max_train_samples:
        train_files = train_files[:config.max_train_samples]
    if config.max_val_samples:
        val_files = val_files[:config.max_val_samples]
    trainset_size, valset_size = len(train_files), len(val_files)
    # an "epoch" can be shorter than a full pass over the training images: the validation,
    # the curves and the prediction plots are then updated more often for the same training time
    steps_per_epoch = config.steps_per_epoch or max(1, trainset_size // config.batch_size)
    validation_steps = max(1, valset_size // config.batch_size)
    print(f"Training on {trainset_size} images, validating on {valset_size} images; "
          f"{steps_per_epoch} batches of {config.batch_size} per epoch "
          f"(= {steps_per_epoch * config.batch_size / trainset_size:.2f} pass over the training images), "
          f"{config.epochs} epochs")

    train_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, train_files)))\
        .map(parse_image, num_parallel_calls=N_CPUS)
    val_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, val_files)))\
        .map(parse_image, num_parallel_calls=N_CPUS)

    # per-pixel loss weights (Keras' `class_weight` is not supported for segmentation targets):
    # the ignored classes "no_data"(0) and "clouds"(1) always get a weight of 0
    counts = LCD.TRAIN_CLASS_COUNTS.astype(np.float64)
    if config.class_weighting is None:
        class_weight = np.ones(LCD.N_CLASSES)
    elif config.class_weighting == 'inverse_sqrt_freq':
        # softer than inverse frequency (which would give a weight > 100 000 to "snow")
        median_count = np.median(counts[2:])
        class_weight = np.sqrt(median_count / np.maximum(counts, 1.))
        class_weight = np.minimum(class_weight, 10.)
    else:
        raise ValueError(f"Unknown class_weighting '{config.class_weighting}'")
    class_weight[LCD.IGNORED_CLASSES_IDX] = 0.
    print(f"Will use class weights: {np.round(class_weight, 3)}")
    class_weight_tf = tf.constant(class_weight, dtype=tf.float32)

    def add_pixel_weights(image, mask):
        weights = tf.gather(class_weight_tf, tf.cast(mask[..., 0], tf.int32))
        return image, mask, weights

    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=N_CPUS)\
        .map(add_pixel_weights, num_parallel_calls=N_CPUS)\
        .shuffle(buffer_size=1024, seed=config.seed)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.map(load_image_test, num_parallel_calls=N_CPUS)\
        .map(add_pixel_weights, num_parallel_calls=N_CPUS)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # where to write files for this experiments
    xp_dir = config.xp_rootdir / (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{config.model.name}"
    )  # no ':' in the name, forbidden on Windows
    (xp_dir/'tensorboard').mkdir(parents=True)
    (xp_dir/'plots').mkdir()
    (xp_dir/'checkpoints').mkdir()
    # save the validation samples to a CSV
    val_samples_s = pd.Series([int(f.stem) for f in val_files], name='sample_id', dtype='uint32')
    val_samples_s.to_csv(xp_dir/'val_samples.csv', index=False)

    # visualize a few samples of the dataset (images, false colours, masks)
    samples_png = plot_dataset_samples(train_files, xp_dir/'plots'/'dataset_samples.png',
                                       num=config.num_dataset_samples)
    print(f"Dataset samples saved to {samples_png}")
    if config.show_plots:
        open_file(samples_png)

    # keep a few fixed validation images to follow the predictions during training
    n_vis = config.num_prediction_samples
    vis_images, vis_masks = [], []
    for image, mask, _ in val_dataset.unbatch().take(n_vis):
        vis_images.append(image.numpy()); vis_masks.append(mask.numpy())
    vis_images, vis_masks = np.stack(vis_images), np.stack(vis_masks)
    vis_callback = VisualizationCallback(vis_images, vis_masks, xp_dir/'plots')

    callbacks = [
        vis_callback,
        tf.keras.callbacks.TensorBoard(
            log_dir=str(xp_dir/'tensorboard'),
            update_freq='epoch'
        ),
        # tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(xp_dir/'checkpoints'/'epoch{epoch}'), save_best_only=False, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(xp_dir/'fit_logs.csv')
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=config.lr_patience,
            factor=0.5,
            verbose=1,
        )
    ]
    # create the model to train (see the "model" section of the YAML config)
    model_kwargs = vars(config.model.kwargs) if getattr(config.model, 'kwargs', None) else {}
    if config.model.name == 'deeplabv3plus':
        # per-band statistics of the training images, to standardize the inputs of the model
        stats_images = np.stack([
            load_image_test(*parse_image(tf.constant(str(f))))[0].numpy()
            for f in train_files[:200]
        ])
        model_kwargs['input_mean'] = stats_images.mean(axis=(0, 1, 2)).tolist()
        model_kwargs['input_variance'] = stats_images.var(axis=(0, 1, 2)).tolist()
        print(f"Input statistics (on {len(stats_images)} images): mean={np.round(model_kwargs['input_mean'], 4)}, "
              f"std={np.round(np.sqrt(model_kwargs['input_variance']), 4)}")
        del stats_images
    print(f"Creating model '{config.model.name}'")
    model = build_model(
        config.model.name,
        input_shape=(LCD.IMG_SIZE, LCD.IMG_SIZE, LCD.N_CHANNELS),
        num_classes=LCD.N_CLASSES,
        **model_kwargs
    )
    model.summary()

    # get optimizer, loss, and compile model for training
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    print("Compile model")
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[PixelAccuracy(), ClassDistributionKL()],
                  weighted_metrics=[])
                  # metrics = [tf.keras.metrics.Precision(),
                  #            tf.keras.metrics.Recall(),
                  #            tf.keras.metrics.MeanIoU(num_classes=LCD.N_CLASSES)]) # TODO segmentation metrics

    # Launch training
    model_history = model.fit(train_dataset, epochs=config.epochs,
                              callbacks=callbacks,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_dataset,
                              validation_steps=validation_steps
                              )
    print(f"Done. Experiment saved in: {xp_dir}")
    print(f"Training curves: {vis_callback.curves_png}")
    print(f"Last predictions: {vis_callback.last_predictions_png}")
    if config.show_plots:
        open_file(vis_callback.curves_png)
        if vis_callback.last_predictions_png is not None:
            open_file(vis_callback.last_predictions_png)

