#!/bin/bash
# Entraînements longs de DeepLabV3+ (init. ImageNet) pour une nuit (~8 h 30 sur RTX 4090).
# Tous les runs : 120 époques, LR cosinus 1e-3 -> 1e-5 (warm-up 2 époques), batch 32,
# poids de classes du groupe, découpage train / val / test du groupe.
#
#   run        loss    augmentation radiométrique   graine   rôle
#   A_s42      ce_kl   oui                          42       config principale
#   A_s1       ce_kl   oui                          1        ensemble
#   B_ce_s42   ce      oui                          42       ablation de la loss (CE seule)
#   A_s2       ce_kl   oui                          2        ensemble
#   C_noaug    ce_kl   non                          42       ablation de l'augmentation radiométrique
#   A_s3       ce_kl   oui                          3        ensemble
# Puis improve_eval.py : ensemble des runs A (+ TTA, température choisie sur la validation)
# et fichier de soumission du challenge de cette configuration.
#
# Après chaque run : évaluation (val + test du groupe + submission.csv) et suppression des
# checkpoints autres que la meilleure époque (place disque).
# L'avancement est dans night_logs/progress.txt (tail -f night_logs/progress.txt).
#
# Usage :  DATASET=/chemin/vers/dataset nohup bash run_night.sh > night.log 2>&1 &
#   EPOCHS=120 (défaut)   DATASET=... (défaut : dataset_folder de train_config.yaml)
export TF_USE_LEGACY_KERAS=1
cd "$(dirname "$0")"
EPOCHS=${EPOCHS:-120}
DATASET=${DATASET:-}
LOG=night_logs
mkdir -p "$LOG"
PROGRESS=$LOG/progress.txt

say () { echo "$(date '+%d/%m %H:%M') - $*" | tee -a "$PROGRESS"; }

# vérifications avant de lancer 8 h de calcul
python -c "import tensorflow as tf, sys; g = tf.config.list_physical_devices('GPU'); print('GPU :', g); sys.exit(0 if g else 1)" \
    || { say "ARRÊT : aucun GPU vu par TensorFlow"; exit 1; }
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
say "Début ($EPOCHS époques par run), espace disque libre : ${FREE_GB} Go"
[ "$FREE_GB" -lt 8 ] && { say "ARRÊT : moins de 8 Go libres"; exit 1; }

# crée la config d'un run à partir de train_config.yaml
make_cfg () {  # $1 fichier, $2 tag, $3 graine, $4 loss (ce | ce_kl), $5 augmentation radiométrique (0 | 1)
python - "$@" "$EPOCHS" "$DATASET" <<'PY'
import sys, yaml
from pathlib import Path
out, tag, seed, loss, radio, epochs, dataset = sys.argv[1:8]
cfg = yaml.safe_load(open('train_config.yaml'))
if dataset:
    cfg['dataset_folder'] = dataset
assert Path(cfg['dataset_folder']).expanduser().is_dir(), f"dataset introuvable : {cfg['dataset_folder']} (donner DATASET=...)"
cfg.update(seed=int(seed), epochs=int(epochs), batch_size=32, lr=1e-3,
           max_train_samples=None, max_val_samples=None, steps_per_epoch=None,
           show_plots=False, val_samples_csv=None, checkpoint_best_only=True, xp_tag=tag)
cfg['model'] = {'name': 'deeplabv3plus', 'kwargs': {'pretrained': 'imagenet', 'backbone_alpha': 1.0}}
cfg['loss'] = {'type': loss, 'ce_weight': 1.0, 'kl_weight': 10.0}
cfg['augmentation'] = {'radiometric': {'prob': 0.8, 'brightness': 0.1, 'contrast': 0.1,
                                       'band_jitter': 0.05, 'noise': 0.03} if radio == '1' else False}
cfg['lr_schedule'] = {'type': 'cosine', 'min_lr': 1.0e-5, 'warmup_epochs': 2}
yaml.safe_dump(cfg, open(out, 'w'), sort_keys=False)
PY
}

ENS_XPS=(); ENS_NAMES=()
run () {  # $1 tag, puis les arguments de make_cfg après le tag
    local tag=$1; shift
    local cfg=$LOG/cfg_$tag.yaml
    make_cfg "$cfg" "$tag" "$@" || { say "ÉCHEC config $tag"; return; }
    say "début $tag"
    python -m framework.train -c "$cfg" > "$LOG/train_$tag.log" 2>&1
    local xp
    xp=$(ls -td experiments/*_"$tag" 2>/dev/null | head -1)
    if [ -z "$xp" ] || [ ! -s "$xp/fit_logs.csv" ]; then
        say "ÉCHEC entraînement $tag (voir $LOG/train_$tag.log)"; return
    fi
    python evaluate_test.py --config "$cfg" --only "$(basename "$xp")" --submission > "$LOG/eval_$tag.log" 2>&1
    # ne garder que le checkpoint de la meilleure époque (choisie sur val_kl)
    python - "$xp" <<'PY'
import sys, shutil, pandas as pd
from pathlib import Path
xp = Path(sys.argv[1])
best = int(pd.read_csv(xp/'fit_logs.csv')['val_kl'].idxmin()) + 1
for d in (xp/'checkpoints').iterdir():
    if d.name != f'epoch{best}':
        shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink()
PY
    say "fin $tag ($(basename "$xp")) | $(grep -m1 'best epoch' "$LOG/eval_$tag.log" | sed 's/.*= //') | $(grep -m1 'test:' "$LOG/eval_$tag.log" | sed 's/^ *//')"
    if [[ $tag == A_* ]]; then ENS_XPS+=("$(basename "$xp")"); ENS_NAMES+=("$tag"); fi
}

run A_s42    42 ce_kl 1
run A_s1      1 ce_kl 1
run B_ce_s42 42 ce    1
run A_s2      2 ce_kl 1
run C_noaug  42 ce_kl 0
run A_s3      3 ce_kl 1

if [ ${#ENS_XPS[@]} -ge 2 ]; then
    say "ensemble + TTA de ${ENS_NAMES[*]}"
    python improve_eval.py --config "$LOG/cfg_A_s42.yaml" --xps "${ENS_XPS[@]}" --names "${ENS_NAMES[@]}" \
        --tag night --submission > "$LOG/improve_eval.log" 2>&1
    say "ensemble : $(grep -m1 'Best configuration' "$LOG/improve_eval.log") | $(grep -m1 'KL divergence (test)' "$LOG/improve_eval.log")"
fi
say "TERMINÉ - résultats : results/summary_test_split.csv et results/summary_improvements_night.csv"
