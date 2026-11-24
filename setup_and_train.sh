#!/bin/bash
# Tout-en-un (Linux) : installation + entraînement + évaluation sur le test du groupe
#   nohup bash setup_and_train.sh /chemin/vers/dataset > train.log 2>&1 &
#   tail -f train.log
# Options : EPOCHS=30 (défaut 40), WITH_UNET=1 (ajoute le U-Net), FORCE_CPU=1 (sans GPU)
cd "$(dirname "$0")"
step () { echo; echo "=== $(date +%H:%M) - $*"; }

DATA="$1"
if [ -z "$DATA" ]; then
  echo "ERREUR : indique le dossier du dataset, par exemple :"
  echo "  nohup bash setup_and_train.sh ~/dataset > train.log 2>&1 &"
  exit 1
fi
DATA="$(cd "$DATA" 2>/dev/null && pwd)"
if [ -z "$DATA" ] || [ ! -d "$DATA/train/images" ] || [ ! -d "$DATA/train/masks" ]; then
  echo "ERREUR : '$1' doit contenir les dossiers train/images et train/masks"
  exit 1
fi
step "Dataset : $DATA ($(ls "$DATA/train/images" | wc -l) images d'entraînement)"
sed -i "s#^dataset_folder: .*#dataset_folder: '$DATA'#" train_config.yaml infer_config.yaml

if ! command -v python3 >/dev/null; then
  echo "ERREUR : python3 n'est pas installé (sudo apt install python3 python3-venv)"; exit 1
fi
if [ ! -f .venv/bin/activate ]; then
  step "Création de l'environnement Python (.venv)"
  rm -rf .venv
  python3 -m venv .venv || { echo "ERREUR : installe venv avec : sudo apt install python3-venv"; exit 1; }
fi
source .venv/bin/activate
export TF_USE_LEGACY_KERAS=1
if ! python -c "import tensorflow, tf_keras, tifffile, yaml, pandas, matplotlib, tqdm" 2>/dev/null; then
  step "Installation des paquets (TensorFlow + CUDA, quelques minutes)"
  pip install --upgrade pip -q
  pip install -q "tensorflow[and-cuda]==2.20.*" "tf-keras==2.20.*" pandas tifffile pyyaml matplotlib tqdm \
    || { echo "ERREUR pendant l'installation des paquets"; exit 1; }
fi

step "Vérification du GPU"
NGPU=$(python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null | tail -1)
echo "GPU détectés : $NGPU"
if [ "$NGPU" = "0" ] && [ "$FORCE_CPU" != "1" ]; then
  echo "ERREUR : aucun GPU utilisable, l'entraînement serait beaucoup trop long."
  echo "Vérifie 'nvidia-smi'. Pour lancer quand même sur CPU : FORCE_CPU=1 bash setup_and_train.sh ..."
  exit 1
fi

step "Entraînement (EPOCHS=${EPOCHS:-40}, WITH_UNET=${WITH_UNET:-0})"
bash run_all.sh

step "Évaluation sur la validation et le test du groupe"
python evaluate_test.py --submission

step "FINI. Résultats dans results/ (tableau : results/summary_test_split.csv)"
