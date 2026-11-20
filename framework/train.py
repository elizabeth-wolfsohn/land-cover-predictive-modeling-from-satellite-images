"""
Train the model.
"""
from pathlib import Path
import datetime
import argparse
import itertools
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


def build_class_weights(counts, mode='power', power=0.5, max_weight=None, ignored_idx=(0, 1)):
    """Poids par classe pour la perte, normalisés pour qu'un pixel pèse 1 en moyenne.
 
    mode :
      - 'uniform'   : poids 1 pour toutes les classes évaluées (aucun rééquilibrage)
      - 'benchmark' : inverse de la fréquence, la formule voulue par le benchmark
                      (équivaut à power=1 sans écrêtage)
      - 'power'     : poids = fréquence ** (-power), puis écrêtage à max_weight
                      (power=0.5 → inverse de la racine, power=1 → inverse)
    Les classes de `ignored_idx` (no_data, clouds) reçoivent toujours un poids nul.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n_classes = len(counts)
    valid = np.array([i not in set(ignored_idx) for i in range(n_classes)])
 
    if mode == 'uniform':
        p, max_weight = 0.0, None
    elif mode == 'benchmark':
        p, max_weight = 1.0, None
    elif mode == 'power':
        p = float(power)
    else:
        raise ValueError(f"class_weighting.mode inconnu : {mode!r} "
                         "(attendu : 'uniform', 'benchmark' ou 'power')")
 
    freq = counts[valid] / counts[valid].sum()
    w_valid = np.zeros_like(freq)
    present = freq > 0                       # une classe absente garde un poids nul
    w_valid[present] = freq[present] ** (-p)
    w_valid /= (w_valid * freq).sum()        # poids moyen par pixel = 1
    if max_weight is not None:
        w_valid = np.minimum(w_valid, float(max_weight))
 
    weights = np.zeros(n_classes, dtype=np.float32)
    weights[valid] = w_valid
    return weights


def kl_proportions(y_true, y_pred):
    true = tf.reduce_mean(tf.one_hot(tf.cast(tf.squeeze(y_true, -1), tf.int32), LCD.N_CLASSES), [1, 2])
    pred = tf.reduce_mean(y_pred, [1, 2])              
    valid = [i for i in range(LCD.N_CLASSES) if i not in LCD.IGNORED_CLASSES_IDX]
    true, pred = tf.gather(true, valid, axis=-1), tf.gather(pred, valid, axis=-1)
    true = true / tf.reduce_sum(true, -1, keepdims=True) + 1e-7
    pred = pred / tf.reduce_sum(pred, -1, keepdims=True) + 1e-7
    return tf.reduce_sum(true * tf.math.log(true / pred), -1)  


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


def _optional_config_value(config, key, default=None):
    """Renvoie config.<key>, ou `default` si la clé est absente du fichier YAML."""
    try:
        value = getattr(config, key)
    except (AttributeError, KeyError):
        return default
    return default if value is None else value


def _optional_file(config, key):
    """Lit une clé de chemin optionnelle et vérifie que le fichier existe."""
    value = _optional_config_value(config, key)
    if value is None:
        return None
    path = Path(value).expanduser()
    assert path.is_file(), f"{key} : fichier introuvable : {path}"
    return path


def _read_ids(csv_path: Path) -> list:
    """Lit un CSV d'une seule colonne d'identifiants d'images (nnnnn), avec ou sans en-tête.

    Renvoie une liste d'entiers. Lève une erreur si une valeur n'est pas numérique
    (hors éventuelle ligne d'en-tête) ou si un identifiant apparaît deux fois.
    """
    col = pd.read_csv(csv_path, header=None, dtype=str).iloc[:, 0].str.strip()
    ids = pd.to_numeric(col, errors='coerce')
    bad = ids.isna()
    # seule la première ligne a le droit d'être non numérique (en-tête, ex. "sample_id")
    if bad.iloc[1:].any():
        raise ValueError(f"{csv_path} : valeurs non numériques : {col[bad].tolist()[:10]}")
    ids = ids[~bad].astype(np.int64)
    duplicated = ids[ids.duplicated()]
    if len(duplicated):
        raise ValueError(f"{csv_path} : identifiants en double : {duplicated.tolist()[:10]}")
    return ids.tolist()


def _ids_to_files(ids: list, files_by_id: dict, csv_path: Path) -> list:
    """Convertit des identifiants en chemins de fichiers .tif, en signalant les absents."""
    missing = [i for i in ids if i not in files_by_id]
    if missing:
        raise FileNotFoundError(
            f"{csv_path} : {len(missing)} image(s) absente(s) de train/images, ex. {missing[:10]}")
    return [files_by_id[i] for i in ids]


def _parse_args():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--config', '-c', type=str, required=True, help="The YAML config file")
    cli_args = parser.parse_args()
    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    config.xp_rootdir = Path(config.xp_rootdir).expanduser()
    assert config.xp_rootdir.is_dir()
    config.dataset_folder = Path(config.dataset_folder).expanduser()
    print(config.dataset_folder)
    assert config.dataset_folder.is_dir()
    # découpage prédéfini train / val / test (optionnel)
    config.train_image_path = _optional_file(config, 'train_image_path')
    config.val_image_path = _optional_file(config, 'val_image_path')
    config.test_image_path = _optional_file(config, 'test_image_path')
    config.evaluate_on_test = bool(_optional_config_value(config, 'evaluate_on_test', False))
    # ancien mode : seule la liste de validation est fournie
    config.val_samples_csv = _optional_file(config, 'val_samples_csv')

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
    # tri pour que l'ordre (et donc le mélange avec la graine) soit reproductible
    all_files = sorted(config.dataset_folder.glob('train/images/*.tif'))
    files_by_id = {int(f.stem): f for f in all_files}
    test_files = []

    if config.train_image_path is not None:
        # --- Découpage prédéfini train / val / test, lu depuis les CSV ---
        assert config.val_image_path is not None, \
            "val_image_path doit être renseigné quand train_image_path l'est"
        split_ids = {
            'train': _read_ids(config.train_image_path),
            'val': _read_ids(config.val_image_path),
            'test': _read_ids(config.test_image_path) if config.test_image_path is not None else [],
        }
        # une image ne doit jamais apparaître dans deux ensembles (fuite de données)
        for (name_a, ids_a), (name_b, ids_b) in itertools.combinations(split_ids.items(), 2):
            overlap = set(ids_a) & set(ids_b)
            if overlap:
                raise ValueError(f"{len(overlap)} image(s) à la fois dans {name_a} et {name_b}, "
                                 f"ex. {sorted(overlap)[:10]}")
        train_files = _ids_to_files(split_ids['train'], files_by_id, config.train_image_path)
        val_files = _ids_to_files(split_ids['val'], files_by_id, config.val_image_path)
        if config.test_image_path is not None:
            test_files = _ids_to_files(split_ids['test'], files_by_id, config.test_image_path)
        # mélange de la liste d'entraînement (reproductible grâce à la graine)
        train_files = random.sample(train_files, len(train_files))
        n_unused = len(files_by_id) - sum(len(ids) for ids in split_ids.values())
        print(f"Split prédéfini : {len(train_files)} train, {len(val_files)} val, "
              f"{len(test_files)} test ({n_unused} image(s) de train/images non utilisée(s))")
    elif config.val_samples_csv is not None:
        # --- Ancien mode : liste de validation fournie, tout le reste en entraînement ---
        val_ids = set(_read_ids(config.val_samples_csv))
        val_files = _ids_to_files(sorted(val_ids), files_by_id, config.val_samples_csv)
        train_files = [f for f in all_files if int(f.stem) not in val_ids]
        train_files = random.sample(train_files, len(train_files))
    else:
        # --- Ancien mode : validation tirée au hasard (10 % des images) ---
        shuffled = random.sample(all_files, len(all_files))
        n_val = int(len(shuffled) * 0.1)
        train_files, val_files = shuffled[n_val:], shuffled[:n_val]

    trainset_size = len(train_files)
    valset_size = len(val_files)

    train_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, train_files)))\
        .map(parse_image, num_parallel_calls=N_CPUS)
    val_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, val_files)))\
        .map(parse_image, num_parallel_calls=N_CPUS)

    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=N_CPUS)\
        .shuffle(buffer_size=1024, seed=config.seed)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.map(load_image_test, num_parallel_calls=N_CPUS)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # where to write files for this experiments
    xp_dir = config.xp_rootdir / datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    (xp_dir/'tensorboard').mkdir(parents=True)
    (xp_dir/'plots').mkdir()
    (xp_dir/'checkpoints').mkdir()
    # sauvegarde des listes utilisées, pour pouvoir reproduire l'expérience
    for split_name, split_files in (('train', train_files), ('val', val_files), ('test', test_files)):
        if split_files:
            pd.Series(sorted(int(f.stem) for f in split_files), name='sample_id', dtype='uint32')\
                .to_csv(xp_dir/f'{split_name}_samples.csv', index=False)

    # keep a training minibatch for visualization
    for image, mask in train_dataset.take(1):
        sample_batch = (image[:5, ...], mask[:5, ...])

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
        input_shape=(LCD.IMG_SIZE, LCD.IMG_SIZE, LCD.N_CHANNELS),
        num_classes=LCD.N_CLASSES,
        num_layers=2
    )
    print(f"Creating U-Net with arguments: {unet_kwargs}")
    model = UNet(**unet_kwargs)
    print(model.summary())

    # get optimizer, loss, and compile model for training
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

    # compute class weights for the loss: inverse-frequency balanced
    cw_cfg = _optional_config_value(config, 'class_weighting')
    class_weight = build_class_weights(
        LCD.TRAIN_CLASS_COUNTS,
        mode=_optional_config_value(cw_cfg, 'mode', 'power'),
        power=_optional_config_value(cw_cfg, 'power', 0.5),
        max_weight=_optional_config_value(cw_cfg, 'max_weight', None),
        ignored_idx=LCD.IGNORED_CLASSES_IDX,
    )
    print(f"Will use class weights: {class_weight}")
    cw = tf.constant(class_weight)

    def add_sample_weights(image, mask):
        # mask : (B, 256, 256, 1) uint8 → poids par pixel : (B, 256, 256)
        weights = tf.gather(cw, tf.cast(tf.squeeze(mask, -1), tf.int32))
        return image, mask, weights

    train_dataset = train_dataset.map(add_sample_weights)
    # -----------------------------------------


    print(f"Will use class weights: {class_weight}")

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    print("Compile model")
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(name="ce"),
                           tf.keras.metrics.MeanMetricWrapper(kl_proportions, name="kl")])
                  # metrics = [tf.keras.metrics.Precision(),
                  #            tf.keras.metrics.Recall(),
                  #            tf.keras.metrics.MeanIoU(num_classes=LCD.N_CLASSES)]) # TODO segmentation metrics

    # Launch training
    model_history = model.fit(train_dataset, epochs=config.epochs,
                              callbacks=callbacks,
                              steps_per_epoch=trainset_size // config.batch_size,
                              validation_data=val_dataset,
                              validation_steps=valset_size // config.batch_size
                              )

    # Évaluation sur le test : désactivée par défaut, pour ne pas régler les
    # hyperparamètres en regardant le test (activer avec evaluate_on_test: true)
    if config.evaluate_on_test and test_files:
        test_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, test_files)))\
            .map(parse_image, num_parallel_calls=N_CPUS)\
            .map(load_image_test, num_parallel_calls=N_CPUS)\
            .batch(config.batch_size)\
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print('Evaluate on the test set (poids de la dernière époque)')
        test_metrics = model.evaluate(test_dataset, return_dict=True)
        print(f"Test : {test_metrics}")
        with open(xp_dir/'test_metrics.yaml', 'w') as f:
            yaml.safe_dump({k: float(v) for k, v in test_metrics.items()}, f)

