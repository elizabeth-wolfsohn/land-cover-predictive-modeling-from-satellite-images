
"""
Train the ResNet50-U-Net model.
"""
from pathlib import Path
import datetime
import argparse
import yaml
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from framework.dataset import LandCoverData as LCD
from framework.dataset import parse_image, load_image_train, load_image_test
from framework.models.resnet50 import build_resnet50_unet
from framework.tensorflow_utils import plot_predictions
from framework.utils import YamlNamespace



class PlotCallback(tf.keras.callbacks.Callback):
    """A callback used to display sample predictions during training."""

    from IPython.display import clear_output

    def __init__(
        self,
        dataset: tf.data.Dataset = None,
        sample_batch: tf.Tensor = None,
        save_folder: Path = None,
        num: int = 1,
        ipython_mode: bool = False
    ):
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
            save_filepaths = [
                self.save_folder / f'plot_{n}_epoch{epoch}.png'
                for n in range(1, self.num + 1)
            ]
        else:
            save_filepaths = None

        plot_predictions(
            self.model,
            self.dataset,
            self.sample_batch,
            num=self.num,
            save_filepaths=save_filepaths
        )


def _optional_config_value(config, key, default=None):
    """Return a configuration value, or default if it is absent."""

    try:
        value = getattr(config, key)
    except (AttributeError, KeyError):
        return default

    return default if value is None else value


def _parse_args():

    parser = argparse.ArgumentParser('Training script')

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help="The YAML config file"
    )

    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = YamlNamespace(config)

    config.xp_rootdir = Path(config.xp_rootdir).expanduser()
    assert config.xp_rootdir.is_dir()

    config.dataset_folder = Path(config.dataset_folder).expanduser()
    assert config.dataset_folder.is_dir()

    # Optional validation CSV
    val_samples_csv = _optional_config_value(
        config, 'val_samples_csv'
    )

    config.val_samples_csv = (
        Path(val_samples_csv).expanduser()
        if val_samples_csv is not None else None
    )

    if config.val_samples_csv is not None:
        assert config.val_samples_csv.is_file()

    # Optional validation folder
    validation_folder = _optional_config_value(
        config, 'validation_folder'
    )

    config.validation_folder = (
        Path(validation_folder).expanduser()
        if validation_folder is not None else None
    )

    if config.validation_folder is not None:
        assert config.validation_folder.is_dir()

    # The two validation methods cannot be used simultaneously
    if (
        config.val_samples_csv is not None
        and config.validation_folder is not None
    ):
        raise ValueError(
            "Choose either val_samples_csv or validation_folder, not both."
        )

    config.seed = _optional_config_value(config, 'seed', None)

    return config


if __name__ == '__main__':

    import multiprocessing

    config = _parse_args()

    print(f'Config:\n{config}')

    # Set random seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    N_CPUS = multiprocessing.cpu_count()

    # ---------------------------------------------------------
    # Class weights (same formula as the reference U-Net)
    # ---------------------------------------------------------

    counts = LCD.TRAIN_CLASS_COUNTS.astype(np.float64)

    valid = [
        c for c in range(LCD.N_CLASSES)
        if c not in LCD.IGNORED_CLASSES_IDX
    ]

    if len(valid) == 0 or np.any(counts[valid] <= 0):
        raise ValueError(
            "Invalid class counts: check LCD.TRAIN_CLASS_COUNTS."
        )

   # class_weight = np.zeros(LCD.N_CLASSES, dtype=np.float32)

    # class_weight[valid] = (
    #     counts[valid].sum()
    #     / (counts[valid] * len(valid))
    # )

    # class_weight = np.minimum(class_weight, 10.)
     # compute class weights for the loss: inverse-frequency balanced
    # note: we set to 0 the weights for the classes "no_data"(0) and "clouds"(1) to ignore these
    class_weight = (1 / LCD.TRAIN_CLASS_COUNTS[2:])* LCD.TRAIN_CLASS_COUNTS[2:].sum() / (LCD.N_CLASSES-2)
    class_weight[LCD.IGNORED_CLASSES_IDX] = 0.

    print(f"Will use class weights: {class_weight}")

    # class_weight_tf = tf.constant(
    #     class_weight,
    #     dtype=tf.float32
    # )

    # def add_sample_weights(image, mask):
    #     # Per-pixel weight depending on the class
    #     weights = tf.gather(
    #         class_weight_tf,
    #         tf.cast(tf.squeeze(mask, axis=-1), tf.int32)
    #     )

    #     return image, mask, weights

    # ---------------------------------------------------------
    # Training and validation datasets
    # ---------------------------------------------------------

    print('Instantiate train and validation datasets')

    train_files = sorted(config.dataset_folder.glob('train/images/*.tif'))

    if not train_files:
        raise ValueError("No training images found.")

    # Shuffle training samples
    train_files = random.sample(train_files, len(train_files))

    devset_size = len(train_files)

    # ---------------------------------------------------------
    # Validation set
    # ---------------------------------------------------------

    if config.validation_folder is not None:

        # Validation images and masks from a separate folder
        val_files = sorted(
            config.validation_folder.glob('images/*.tif')
        )

        if not val_files:
            raise ValueError(
                f"No validation images found in "
                f"{config.validation_folder / 'images'}"
            )

        # Check that every validation image has a corresponding mask
        missing_masks = [
            f.name for f in val_files
            if not (
                config.validation_folder / 'masks' / f.name
            ).is_file()
        ]

        if missing_masks:
            raise FileNotFoundError(
                f"Missing validation masks, examples: "
                f"{missing_masks[:10]}"
            )

        valset_size = len(val_files)
        trainset_size = len(train_files)

        print(
            f"Using external validation folder: "
            f"{trainset_size} train, {valset_size} validation"
        )

    elif config.val_samples_csv is not None:

        # Read validation sample IDs from CSV
        val_samples = pd.read_csv(
            config.val_samples_csv
        ).iloc[:, 0].astype(int).tolist()

        val_ids = set(val_samples)

        if len(val_ids) != len(val_samples):
            raise ValueError(
                "Duplicate IDs in validation CSV."
            )

        files_by_id = {
            int(f.stem): f for f in train_files
        }

        missing_ids = [
            i for i in val_samples
            if i not in files_by_id
        ]

        if missing_ids:
            raise FileNotFoundError(
                f"Validation IDs not found: {missing_ids[:10]}"
            )

        val_files = [
            files_by_id[i] for i in val_samples
        ]

        train_files = [
            f for f in train_files
            if int(f.stem) not in val_ids
        ]

        valset_size = len(val_files)
        trainset_size = len(train_files)

        assert valset_size + trainset_size == devset_size

    else:

        # Generate a random 10% hold-out validation set
        valset_size = int(len(train_files) * 0.1)

        if valset_size == 0:
            raise ValueError(
                "Not enough images to create a validation set."
            )

        train_files, val_files = (
            train_files[valset_size:],
            train_files[:valset_size]
        )

        trainset_size = len(train_files)

    print(
        f"Dataset sizes: {trainset_size} train, "
        f"{valset_size} validation"
    )

    # ---------------------------------------------------------
    # Build TensorFlow datasets
    # ---------------------------------------------------------

    train_dataset = tf.data.Dataset.from_tensor_slices(
        list(map(str, train_files))
    ).map(
        parse_image,
        num_parallel_calls=N_CPUS
    )

    val_dataset = tf.data.Dataset.from_tensor_slices(
        list(map(str, val_files))
    ).map(
        parse_image,
        num_parallel_calls=N_CPUS
    )

    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=N_CPUS)\
            .shuffle(buffer_size=1024, seed=config.seed)\
            .repeat()\
            .batch(config.batch_size)\
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.map(load_image_test, num_parallel_calls=N_CPUS)\
           .repeat()\
           .batch(config.batch_size)\
           .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # ---------------------------------------------------------
    # Experiment directory
    # ---------------------------------------------------------

    xp_dir = config.xp_rootdir / datetime.datetime.now().strftime(
        "%d-%m-%Y_%H-%M-%S"
    )

    (xp_dir / 'tensorboard').mkdir(parents=True)
    (xp_dir / 'plots').mkdir()
    (xp_dir / 'checkpoints').mkdir()

    # Save validation sample IDs
    val_samples_s = pd.Series(
        [int(f.stem) for f in val_files],
        name='sample_id',
        dtype='uint32'
    )

    val_samples_s.to_csv(
        xp_dir / 'val_samples.csv',
        index=False
    )

    # Save training sample IDs
    train_samples_s = pd.Series(
        [int(f.stem) for f in train_files],
        name='sample_id',
        dtype='uint32'
    )

    train_samples_s.to_csv(
        xp_dir / 'train_samples.csv',
        index=False
    )

    # ---------------------------------------------------------
    # Visualization batch
    # ---------------------------------------------------------

    for image, mask in train_dataset.take(1):
        sample_batch = (
            image[:5, ...],
            mask[:5, ...]
        )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    callbacks = [

        PlotCallback(
            sample_batch=sample_batch,
            save_folder=xp_dir / 'plots',
            num=5
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=xp_dir / 'tensorboard',
            update_freq='epoch'
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(xp_dir / 'checkpoints' / 'epoch{epoch}'),
            save_best_only=False,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            filename=xp_dir / 'fit_logs.csv'
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            patience=20,
            factor=0.5,
            verbose=1
        )
    ]

    # ---------------------------------------------------------
    # Create the ResNet50-U-Net model
    # ---------------------------------------------------------

    resnet_kwargs = dict(
        input_shape=(
            LCD.IMG_SIZE,
            LCD.IMG_SIZE,
            LCD.N_CHANNELS
        ),
        num_classes=LCD.N_CLASSES,
        output_activation="softmax"
    )

    print(
        f"Creating ResNet50-U-Net with arguments: {resnet_kwargs}"
    )

    model = build_resnet50_unet(**resnet_kwargs)

    model.summary()

    # ---------------------------------------------------------
    # Compile model
    # ---------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.lr
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    )

    print("Compile model")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[]
        # metrics = [tf.keras.metrics.Precision(),
        #            tf.keras.metrics.Recall(),
        #            tf.keras.metrics.MeanIoU(num_classes=LCD.N_CLASSES)]) # TODO segmentation metrics
        
    )

    # ---------------------------------------------------------
    # Launch training
    # ---------------------------------------------------------

    if trainset_size == 0 or valset_size == 0:
        raise ValueError(
            "Training and validation datasets must be non-empty."
        )

    

    model_history = model.fit(
        train_dataset,
        epochs=config.epochs,
        callbacks=callbacks,
        steps_per_epoch=trainset_size // config.batch_size,
        validation_data=val_dataset,
        validation_steps=valset_size // config.batch_size,
        class_weight=class_weight
    )

