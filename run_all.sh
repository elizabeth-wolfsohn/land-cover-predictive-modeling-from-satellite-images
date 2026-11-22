#!/bin/bash
# Entraîne les modèles à la suite avec le découpage train / validation / test du groupe
# (train_image_path, val_image_path, test_image_path dans train_config.yaml).
# Durée sur GPU : environ 45 s par époque pour DeepLabV3+.
#   EPOCHS=40 (défaut)      -> environ 30 min par modèle DeepLabV3+
#   WITH_UNET=1             -> ajoute le U-Net de référence (plus lent)
# Exemple : EPOCHS=30 WITH_UNET=1 bash run_all.sh
export TF_USE_LEGACY_KERAS=1
cd "$(dirname "$0")"
EPOCHS=${EPOCHS:-40}
WITH_UNET=${WITH_UNET:-0}

# crée un YAML à partir de train_config.yaml avec le modèle voulu
make_cfg () {  # $1 = fichier de sortie, $2 = modèle, $3 = pretrained
python - "$@" "$EPOCHS" <<'PY'
import sys, yaml
out, name, pretrained, epochs = sys.argv[1:5]
cfg = yaml.safe_load(open('train_config.yaml'))
cfg.update(epochs=int(epochs), batch_size=32, max_train_samples=None, max_val_samples=None,
           steps_per_epoch=None, lr_patience=5, show_plots=False, val_samples_csv=None)
cfg['model'] = ({'name': 'unet', 'kwargs': {'num_layers': 2}} if name == 'unet' else
                {'name': 'deeplabv3plus',
                 'kwargs': {'pretrained': None if pretrained == 'none' else pretrained,
                            'backbone_alpha': 1.0}})
yaml.safe_dump(cfg, open(out, 'w'), sort_keys=False)
PY
}

echo "$(date +%H:%M) - $EPOCHS époques par modèle"

# 1) DeepLabV3+ from scratch
make_cfg cfg_1.yaml deeplabv3plus none
python -m framework.train -c cfg_1.yaml > log_1_deeplab_scratch.txt 2>&1
echo "$(date +%H:%M) - modèle 1 terminé (DeepLabV3+ from scratch)"

# 2) DeepLabV3+ pré-entraîné ImageNet
make_cfg cfg_2.yaml deeplabv3plus imagenet
python -m framework.train -c cfg_2.yaml > log_2_deeplab_imagenet.txt 2>&1
echo "$(date +%H:%M) - modèle 2 terminé (DeepLabV3+ ImageNet)"

# 3) U-Net de référence (optionnel)
if [ "$WITH_UNET" = "1" ]; then
  make_cfg cfg_3.yaml unet none
  python -m framework.train -c cfg_3.yaml > log_3_unet.txt 2>&1
  echo "$(date +%H:%M) - modèle 3 terminé (U-Net)"
fi

echo "Terminé"
