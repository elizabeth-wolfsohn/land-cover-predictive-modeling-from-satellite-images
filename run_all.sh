#!/bin/bash
export TF_USE_LEGACY_KERAS=1
cd "$(dirname "$0")"

# crée un YAML à partir de train_config.yaml avec le modèle voulu
make_cfg () {  # $1 = fichier de sortie, $2 = modèle, $3 = pretrained, $4 = val_samples_csv
python - "$@" <<'PY'
import sys, yaml
out, name, pretrained, val_csv = sys.argv[1:5]
cfg = yaml.safe_load(open('train_config.yaml'))
cfg.update(epochs=60, batch_size=32, max_train_samples=None, max_val_samples=None,
           steps_per_epoch=None, lr_patience=5, show_plots=False,
           val_samples_csv=None if val_csv == 'none' else val_csv)
cfg['model'] = ({'name': 'unet', 'kwargs': {'num_layers': 2}} if name == 'unet' else
                {'name': 'deeplabv3plus',
                 'kwargs': {'pretrained': None if pretrained == 'none' else pretrained,
                            'backbone_alpha': 1.0}})
yaml.safe_dump(cfg, open(out, 'w'), sort_keys=False)
PY
}

# 1) DeepLabV3+ from scratch (crée le découpage de validation)
make_cfg cfg_1.yaml deeplabv3plus none none
python -m framework.train -c cfg_1.yaml > log_1_deeplab_scratch.txt 2>&1
VAL="$(ls -td experiments/*/ | head -1)val_samples.csv"
echo "Validation commune : $VAL"

# 2) DeepLabV3+ pré-entraîné ImageNet, même validation
make_cfg cfg_2.yaml deeplabv3plus imagenet "$VAL"
python -m framework.train -c cfg_2.yaml > log_2_deeplab_imagenet.txt 2>&1

# 3) U-Net benchmark, même validation
make_cfg cfg_3.yaml unet none "$VAL"
python -m framework.train -c cfg_3.yaml > log_3_unet.txt 2>&1

echo "Terminé"
