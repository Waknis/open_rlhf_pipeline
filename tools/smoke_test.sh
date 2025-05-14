#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define directories for smoke test artifacts to keep them separate
SMOKE_DATA_DIR="data/_smoke_test_data"
SMOKE_PROCESSED_DIR="data/_smoke_test_processed"
SMOKE_MODELS_DIR="models/_smoke_test_models"
SFT_MODEL_PATH="${SMOKE_MODELS_DIR}/sft"
RM_MODEL_PATH="${SMOKE_MODELS_DIR}/rm"
PPO_MODEL_PATH="${SMOKE_MODELS_DIR}/ppo"

# Cleanup function for trap
cleanup() {
    echo "Cleaning up smoke test artifacts..."
    rm -rf "${SMOKE_DATA_DIR}"
    rm -rf "${SMOKE_PROCESSED_DIR}"
    rm -rf "${SMOKE_MODELS_DIR}"
    echo "Cleanup complete."
}

# Trap EXIT signal to run cleanup
trap cleanup EXIT

echo "===== Starting Smoke Test ====="

# 1. Create toy data if missing
echo "--- Creating toy data ---"
mkdir -p "${SMOKE_DATA_DIR}"
# Toy instruction data (2 lines)
echo '{"instruction": "What is a smoke test?", "input": "", "output": "A smoke test is a preliminary test to reveal simple failures severe enough to reject a prospective software release."}' > "${SMOKE_DATA_DIR}/instruct_toy.jsonl"
echo '{"instruction": "Explain the concept of a shell script.", "input": "", "output": "A shell script is a computer program designed to be run by a Unix shell, a command-line interpreter."}' >> "${SMOKE_DATA_DIR}/instruct_toy.jsonl"

# Toy preference data (2 lines) - using "text" key
echo '{"text": "Favorite programming language?", "chosen": "Python, for its readability and versatility.", "rejected": "Assembly, too low-level for daily tasks."}' > "${SMOKE_DATA_DIR}/pairs_toy.jsonl"
echo '{"text": "Best way to learn coding?", "chosen": "Consistent practice and building projects.", "rejected": "Just reading books without trying code."}' >> "${SMOKE_DATA_DIR}/pairs_toy.jsonl"
echo "Toy data created in ${SMOKE_DATA_DIR}"

# 2. Run prepare_data
echo "--- Running prepare_data.py ---"
python scripts/prepare_data.py \
    --instruct_data "${SMOKE_DATA_DIR}/instruct_toy.jsonl" \
    --preference_data "${SMOKE_DATA_DIR}/pairs_toy.jsonl" \
    --out_dir "${SMOKE_PROCESSED_DIR}"
echo "prepare_data.py completed."

# 3. Run train_sft
echo "--- Running train_sft.py ---"
python scripts/train_sft.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data "${SMOKE_PROCESSED_DIR}/instruct.jsonl" \
    --output_dir "${SFT_MODEL_PATH}" \
    --max_steps 2 \
    --bf16 # Assuming CPU run, bf16 will fallback to fp32 if no CUDA
echo "train_sft.py completed."

# 4. Run train_reward
echo "--- Running train_reward.py ---"
python scripts/train_reward.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data "${SMOKE_PROCESSED_DIR}/pairs.jsonl" \
    --output_dir "${RM_MODEL_PATH}" \
    --max_steps 2 \
    --bf16
echo "train_reward.py completed."

# 5. Run train_ppo
echo "--- Running train_ppo.py ---"
# Note: PPO requires SFT model as policy and RM model for reward
python scripts/train_ppo.py \
    --policy "${SFT_MODEL_PATH}" \
    --reward_model_path "${RM_MODEL_PATH}" \
    --prompts_data "${SMOKE_PROCESSED_DIR}/instruct.jsonl" \
    --output_dir "${PPO_MODEL_PATH}" \
    --steps 1 \
    --batch_size 1 \
    --mini_batch_size 1 \
    --bf16
echo "train_ppo.py completed."

# 6. Run eval (basic, assuming default bench)
echo "--- Running run_eval.py ---"
# Assuming run_eval.py can take the PPO model and a default benchmark set
# The README example uses 'all', which might be too much for a quick smoke test.
# Let's assume for now it has a quick default if --bench is not specified, or we use a minimal one.
# For now, let's call it without a specific bench for smoke. Or, if there is a toy bench, that would be better.
# For now, let's use a placeholder that it's being called. Actual benchmarks might need more setup.
if [ -f eval/run_eval.py ]; then
    python eval/run_eval.py --model "${PPO_MODEL_PATH}" # Add --bench if a minimal one exists
    echo "run_eval.py completed."
else
    echo "Skipping run_eval.py (not found)."
fi


# 7. Run safety (basic)
echo "--- Running run_safety.py ---"
# Assuming run_safety.py can take the PPO model
if [ -f eval/run_safety.py ]; then
    python eval/run_safety.py --model "${PPO_MODEL_PATH}"
    echo "run_safety.py completed."
else
    echo "Skipping run_safety.py (not found)."
fi


# 8. Verify model directories exist as a basic check
echo "--- Verifying model outputs ---"
if [ ! -d "${SFT_MODEL_PATH}" ]; then echo "SFT model dir ${SFT_MODEL_PATH} not found!" && exit 1; fi
if [ ! -d "${RM_MODEL_PATH}" ]; then echo "RM model dir ${RM_MODEL_PATH} not found!" && exit 1; fi
if [ ! -d "${PPO_MODEL_PATH}" ]; then echo "PPO model dir ${PPO_MODEL_PATH} not found!" && exit 1; fi
echo "Model directories found."

echo "===== SMOKE OK ====="
exit 0 