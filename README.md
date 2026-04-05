# Open RLHF Pipeline

A reproducible RLHF baseline that fine-tunes a causal LM into a chat assistant with supervised fine-tuning (SFT), reward modeling (RM), and alignment via PPO or DPO.

This repo now targets a concrete training-correctness baseline:

- One validated Transformers/TRL trainer stack, pinned in `requirements.txt` and `constraints.txt`
- Canonical processed data contracts for instruction and preference training
- Shared chat-template-aware prompt formatting across training, evaluation, and the demo
- CPU smoke coverage for data prep, SFT, RM, DPO, PPO, sample evaluation, and safety evaluation

## Repository Layout

```text
open_rlhf_pipeline/
├── open_rlhf/            # shared runtime, formatting, and data helpers
├── scripts/              # data prep + training (SFT, RM, PPO, DPO)
├── eval/                 # sample generation + safety smoke evaluation
├── interpret/            # interpretability tool scripts
├── kernels/              # custom Triton / CUDA kernel examples
├── web_demo/             # Gradio chat UI
├── notebooks/            # experimental notebooks
├── config/               # YAML config presets for different hardware
├── tests/                # pytest smoke tests
├── constraints.txt       # exact validated dependency pins
├── requirements.txt      # install entrypoint for the validated stack
└── .github/workflows/ci.yml
```

## Quick Start

**Important Notes Before Starting:**
*   **TRL Version:** This pipeline uses `trl>=0.8.6` as defined in `requirements.txt`. The PPO training script (`scripts/train_ppo.py`) is compatible with this version.
*   **Dataset Acquisition:** The example commands below use placeholder dataset names (`UltraChat.jsonl`, `OpenHermes_pairs.jsonl`). You will need to procure or create your own datasets in the expected formats. The `scripts/prepare_data.py` script processes data from `data/` into `data/processed/`. Ensure your raw data (e.g., `UltraChat.jsonl`, `OpenHermes_pairs.jsonl`) is placed in the `data/` directory before running `prepare_data.py`.
*   **Model Quality with Quick Start:** The quick start commands use minimal training steps (`--max_steps 2` or `--steps 2`). This is for rapidly testing the pipeline functionality and **will not produce a high-quality, usable chat model. Expect repetitive or nonsensical output from the web demo if using these minimal settings. For a functional model, significantly more training data and steps are required.**
*   **Base Model:** The SFT script defaults to `tiny_llama` if no `--model` argument is provided. You can specify other Hugging Face model IDs or local paths.

```bash
git clone https://github.com/your-handle/open_rlhf_pipeline.git
cd open_rlhf_pipeline

pip install -r requirements.txt

# 1 - Prepare canonical data
python scripts/prepare_data.py \
  --instruct_data data/UltraChat.jsonl \
  --preference_data data/OpenHermes_pairs.jsonl

# 2 - Supervised fine-tuning
python scripts/train_sft.py \
  --model sshleifer/tiny-gpt2 \
  --output_dir models/sft \
  --max_steps 2

# 3 - Reward modeling
python scripts/train_reward.py \
  --model models/sft \
  --output_dir models/rm \
  --max_steps 2

# 4A - PPO alignment
python scripts/train_ppo.py \
  --policy models/sft \
  --reward_model_path models/rm \
  --output_dir models/ppo \
  --steps 2

# 4B - DPO alignment
python scripts/train_dpo.py \
  --policy models/sft \
  --output_dir models/dpo \
  --steps 2

# 5 - Sample generation smoke evaluation
python eval/run_eval.py --model models/ppo

# 6 - Safety smoke evaluation
python eval/run_safety.py --model models/ppo --scorer keyword

# 7 - Chat demo
python web_demo/app.py --model models/ppo
```

## Validated Interfaces

- Processed instruction rows: `prompt`, `response`, `text`
- Processed preference rows: `prompt`, `chosen`, `rejected`
- Shared optional runtime flags on train/eval scripts:
  - `--device {auto,cpu,cuda,mps}`
  - `--dtype {auto,float32,float16,bfloat16}`
  - `--seed <int>`
- Eval scripts accept canonical `--model` and the deprecated alias `--model_path`

## Current Scope

Implemented:

- SFT with response-only loss masking on canonical prompt/response data
- Reward modeling on canonical prompt/chosen/rejected data
- DPO on canonical preference pairs with shared prompt formatting
- PPO on canonical prompt data using the validated TRL PPO trainer API
- Sample-generation and safety smoke evaluation scripts
- Pytest smoke coverage wired into CI

Not implemented yet:

- Real benchmark integrations such as MT-Bench, AlpacaEval, MMLU, or HELM
- Full red-team or policy safety evaluation suites
- LoRA/QLoRA and multi-GPU optimization paths

## Contributing

Pull requests that improve correctness, reproducibility, data adapters, evaluation quality, or test coverage are welcome.

## License

Apache-2.0
