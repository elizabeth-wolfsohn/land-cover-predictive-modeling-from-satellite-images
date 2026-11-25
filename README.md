# ENS Challenge Data 2021 : Land cover predictive modeling from satellite images

Code for the Challenge Data competition [“Land cover predictive modeling from satellite images”](https://challengedata.ens.fr/challenges/48) provided by Preligens: a Swin Transformer V2 encoder pretrained on ImageNet with a light FPN decoder, trained for semantic segmentation, in PyTorch (port of the group's TensorFlow pipeline).

## Data

Download the data from the challenge page and unzip it in `dataset_UNZIPPED/`:
```
dataset_UNZIPPED
├── test
│   └── images     (5 043 files, official test set, no masks)
└── train
    ├── images     (18 491 files)
    └── masks      (18 491 files)
```
- Images: Sentinel-2, 16-bit GeoTIFF of size (256, 256, 4), bands R, G, B, NIR, at 10 m per pixel. They are divided by 24356 (maximum pixel value of the training set) when loaded.
- Masks: 8-bit GeoTIFF of size (256, 256), one of 10 classes per pixel: no_data, clouds, artificial, cultivated, broadleaf, coniferous, herbaceous, natural, snow, water. “no_data” and “clouds” are ignored (weight 0 in the loss, never predicted).
- Target of the challenge: the proportion of each class in an image, evaluated with the Kullback-Leibler divergence. Every sample is identified by its `sample_id` (the file name) in the CSVs.

The group's split of `dataset_UNZIPPED/train` (train / validation / test: 14 794 / 1 848 / 1 849 images) is in `local_dataset/csv_files/`. Build the `local_dataset/<set>/{images,masks}` folders once (symbolic links, no copy):
```
python -m framework.make_local_dataset
```

## Installation

```
uv sync
```

## Usage

Every script takes a YAML config file (parameters described in the comments of the files).
```
python -m framework.train -c train_config.yaml       # training -> code/experiments/<xp>
python -m framework.infer -c infer_config.yaml       # predictions (set: 'test' or 'final_test')
python -m framework.eval  -c eval_config.yaml        # KL divergence + confusion matrix (set: 'test')
python -m framework.plot_logs code/experiments/<xp>  # training curves
```
- `infer.py` with `set: 'final_test'` gives the CSV to submit: `code/experiments/<xp>/best_final_test_predicted.csv`.
- `eval.py` needs `infer.py` to be run first on the same set.

## Results

Files shared in `results/<model>/`. Internal test set (1 849 images), `pred_mode: soft`:

| Backbone | KL divergence |
|---|---|
| `swinv2_tiny_window8_256` | 0.00200 |
| `swinv2_tiny_window16_256` | 0.00249 |
| `swinv2_small_window8_256` | 0.00151 |

Public score (official test set) of `swinv2_tiny_window8_256`: 0.0452.