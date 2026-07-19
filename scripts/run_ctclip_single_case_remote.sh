#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/summer_zhl}"
OUT="$ROOT/artifacts/single_case"
mkdir -p "$OUT"

export CTCLIP_TEXT_MODEL_DIR="$ROOT/models/cxrbert"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

date -Is > "$OUT/started_at.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu_info.csv"
"$ROOT/conda_env/bin/python" -c 'import sys, torch, transformers; print(f"python={sys.version}"); print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}")' > "$OUT/environment.txt"
sha256sum "$ROOT/models/ctclip/CT-CLIP_v2.pt" "$ROOT/models/cxrbert/vocab.txt" "$ROOT/data/valid_1_a_1.nii.gz" > "$OUT/input_sha256.txt"
start=$(date +%s)

"$ROOT/conda_env/bin/python" "$ROOT/scripts/ctclip_worker.py" \
  --volume "$ROOT/data/valid_1_a_1.nii.gz" \
  --checkpoint "$ROOT/models/ctclip/CT-CLIP_v2.pt" \
  --source-dir "$ROOT/external/CT-CLIP-main" \
  --device cuda:0 \
  --fp16 > "$OUT/predictions.json" 2> "$OUT/inference.log" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  date -Is >> "$OUT/gpu_memory.csv"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$OUT/gpu_memory.csv" || true
  sleep 1
done

wait "$pid"
end=$(date +%s)
echo $((end - start)) > "$OUT/wall_seconds.txt"
date -Is > "$OUT/completed_at.txt"
