#!/bin/bash

INPUT_FILE="weights.txt"
GPU_DEVICES=(0 1)
GPU_INDEX=0

LOG_DIR="logs"
mkdir -p "$LOG_DIR"  # 로그 디렉토리 생성 (없으면)

while IFS= read -r line
do
  WEIGHT_PATH="${line}/best_nll_model.pth"
  GPU=${GPU_DEVICES[$GPU_INDEX]}

  # 로그 파일 이름 만들기 (파일 경로에서 불가능한 문자 제거)
  SAFE_NAME=$(echo "$line" | tr '/' '_' | tr '.' '_')
  LOG_FILE="${LOG_DIR}/${SAFE_NAME}.log"

  echo "Launching on GPU $GPU with weight: $WEIGHT_PATH -> log: $LOG_FILE"

  # 병렬 실행 + 로그 저장
  CUDA_VISIBLE_DEVICES=$GPU python3 test.py \
    --type uni --model resnet20  --data cifar10 --weight "$WEIGHT_PATH" \
    > "$LOG_FILE" 2>&1 &

  GPU_INDEX=$((1 - GPU_INDEX))
  sleep 1
done < "$INPUT_FILE"

wait
echo "All processes completed. Check logs in '$LOG_DIR/'"
