# Open RLHF Pipeline

A reproducible RLHF baseline that fine-tunes a causal LM into a chat assistant with supervised fine-tuning (SFT), reward modeling (RM), and alignment via PPO or DPO.

This repo now targets a concrete training-correctness baseline:

- One validated Transformers/TRL trainer stack, pinned in `requirements.txt` and `constraints.txt`
- Canonical processed data contracts for instruction and preference training
- Shared chat-template-aware prompt formatting across training, evaluation, and the demo
- CPU tests for data prep, SFT, RM, DPO, PPO, held-out reward evaluation, sample generation, and safety evaluation. The tests build a small GPT-2 checkpoint locally.

## Repository Layout

```text
open_rlhf_pipeline/
├── open_rlhf/            # shared runtime, formatting, and data helpers
├── scripts/              # data prep + training (SFT, RM, PPO, DPO)
├── eval/                 # reward-pair ranking, sample generation, and safety evaluation
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
*   Install the pinned trainer stack from `requirements.txt`. The validated PPO path uses TRL 0.12.2.
*   **Dataset Acquisition:** The example commands below use placeholder dataset names (`UltraChat.jsonl`, `OpenHermes_pairs.jsonl`). You will need to procure or create your own datasets in the expected formats. The `scripts/prepare_data.py` script processes data from `data/` into `data/processed/`. Ensure your raw data (e.g., `UltraChat.jsonl`, `OpenHermes_pairs.jsonl`) is placed in the `data/` directory before running `prepare_data.py`.
*   **Model Quality with Quick Start:** The quick start commands use minimal training steps (`--max_steps 2` or `--steps 2`). This is for rapidly testing the pipeline functionality and **will not produce a high-quality, usable chat model. Expect repetitive or nonsensical output from the web demo if using these minimal settings. For a functional model, significantly more training data and steps are required.**
*   SFT defaults to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. You can specify another Hugging Face model ID or a local checkpoint.

```bash
git clone https://github.com/Waknis/open_rlhf_pipeline.git
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

## Evaluate the reward model

Reserve preference pairs before training and keep their prompts out of the training file. Pass the held-out JSONL to the evaluator. Each row contains `prompt`, `chosen`, and `rejected`.

```bash
python eval/run_reward_eval.py --model models/rm \
  --data data/held_out_pairs.jsonl --output_file eval/reports/reward.json
```

Preference accuracy is the fraction of scored pairs where the chosen response receives a strictly higher reward. Ties count as zero wins and are reported separately. The JSON includes each scored pair’s zero-based input index, both rewards, and their difference (chosen minus rejected). Set `--max_length` to the limit used in reward training (default 512). Both stages skip a pair if either transcript exceeds this length. The report includes the skipped count.

## Run the CPU tests

After installing the dependencies, run `bash tools/smoke_test.sh`. It builds the model, tokenizer, and sample datasets in a temporary directory, then exercises the training and evaluation paths with Hugging Face downloads disabled.

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

- SFT with response-only loss masking on canonical prompt/response data, including when a long prompt must be shortened
- Reward modeling on canonical prompt/chosen/rejected data
- DPO on canonical preference pairs with shared prompt formatting
- PPO on canonical prompt data using the validated TRL PPO trainer API
- Held-out reward-model ranking, sample-generation and safety smoke evaluation scripts
- Pytest smoke coverage wired into CI

Not implemented yet:

- Real benchmark integrations such as MT-Bench, AlpacaEval, MMLU, or HELM
- Full red-team or policy safety evaluation suites
- LoRA/QLoRA and multi-GPU optimization paths

## Contributing

Pull requests that improve correctness, reproducibility, data adapters, evaluation quality, or test coverage are welcome.

## License

Apache-2.0
