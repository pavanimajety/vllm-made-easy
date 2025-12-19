#!/bin/bash
set +x

INPUT_LEN=1024
OUTPUT_LEN=1024
NUM_PROMPTS=512
CONC=2048
PORT=8000
HOST=0.0.0.0
MODEL=/tmp/nvidia-DeepSeek-R1-FP4-v2

# Wait for server to be ready
echo "Waiting for server to be ready..."
while ! curl -s http://${HOST}:${PORT}/health > /dev/null; do
  echo "Server not ready, waiting 5 seconds..."
  sleep 5
done
echo "Server is ready!"

vllm bench serve \
  --backend vllm \
  --host ${HOST} \
  --port ${PORT} \
  --model ${MODEL} \
  --num-prompts ${NUM_PROMPTS} \
  --dataset-name random \
  --random-input ${INPUT_LEN} \
  --random-output ${OUTPUT_LEN} \
  --max-concurrency ${CONC} \
  --profile
