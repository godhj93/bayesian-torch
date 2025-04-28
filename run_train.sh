#!/usr/bin/env bash
#
# run_pruned_models_parallel.sh
# Description: Run pruned and original models on GPUs using round-robin parallelism.

# ===============================
# Configurable Variables
# ===============================
DATASET="cifar100"
MODEL="resnet20"
OPTIMIZER="sgd"
MOMENTUM=0.9
WEIGHT_DECAY=0.0
STD=1e-2
LR=0.01
EPOCHS=300
BATCH_SIZE=128

BASE_DIR="/home/hj/Projects/bayesian-torch/runs/${DATASET}/${MODEL}/20250428/dnn/bs_${BATCH_SIZE}_opt_${OPTIMIZER}_momentum_${MOMENTUM}_weight_decay_${WEIGHT_DECAY}_nesterov_False_lr_${LR}_mc_runs_${MC_RUNS}_epochs_${EPOCHS}_moped_${MOPED}_timestamp_20250428-221546"
SCRIPT="python3 train_with_good_prior.py --type uni --model ${MODEL} --data ${DATASET} --optimizer ${OPTIMIZER} --momentum ${MOMENTUM} --weight_decay ${WEIGHT_DECAY} --std ${STD} --weight"

ITERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
GPUS=(0 1)

# ===============================
# Launch Pruned Models
# ===============================
echo "Launching ${#ITERS[@]} pruned jobs on GPUs: ${GPUS[*]}"

for idx in "${!ITERS[@]}"; do
  iter=${ITERS[$idx]}
  dev=${GPUS[$(( idx % ${#GPUS[@]} ))]}

  echo "[Iter $iter] → GPU $dev"
  CUDA_VISIBLE_DEVICES=$dev \
    $SCRIPT "${BASE_DIR}/pruned_model_iter_${iter}.pth" &
  # run in background
  sleep 0.2  # small delay to avoid overload

done

# ===============================
# Launch Original Models
# ===============================
echo "[Original A] → GPU 0"
CUDA_VISIBLE_DEVICES=0 $SCRIPT "${BASE_DIR}/original_model.pth" &

echo "[Original B] → GPU 1"
CUDA_VISIBLE_DEVICES=1 $SCRIPT "${BASE_DIR}/original_model.pth" --MOPED &

echo "Baseline Model N(0, 1) from SCRATCH"
CUDA_VISIBLE_DEVICES=0 python3 train.py --type uni --model ${MODEL} --data ${DATASET} --optimizer ${OPTIMIZER} --momentum ${MOMENTUM} --weight_decay ${WEIGHT_DECAY} --std ${STD} &
# ===============================
# Wait for All Background Jobs
# ===============================
wait

echo "✅ All jobs finished."
