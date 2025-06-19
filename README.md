# Open RLHF Pipeline

A fully reproducible pipeline that fine-tunes a Llama-style model into a chat assistant using supervised fine-tuning (SFT), reward modeling, and policy optimization (PPO **or** DPO).
*Runs on one consumer GPU yet scales to multi-GPU clusters.*
*Note: The current `train_ppo.py` script has been adapted for compatibility with TRL `0.8.x` as specified in `requirements.txt`. Ensure your environment matches.*

---
## Repository layout
```text
open_rlhf_pipeline/
├── scripts/               # data prep + training (SFT, RM, PPO, DPO)
├── eval/                  # evaluation + safety scripts
├── interpret/             # interpretability tool scripts
├── kernels/               # custom Triton / CUDA kernel examples
├── web_demo/              # Gradio chat UI
├── notebooks/             # experimental notebooks (e.g., scaling laws)
├── config/                # YAML config presets for different hardware
├── data/                  # (Conventionally) for storing datasets
├── models/                # (Conventionally) for storing trained model checkpoints
├── tests/                 # (Placeholder) for future unit/integration tests
├── requirements.txt       # Python package dependencies
├── Dockerfile             # Docker configuration
└── .github/workflows/ci.yml # CI workflow examples
```

## Quick Start (Example with RTX 4090-level GPU)

**Important Notes Before Starting:**
*   **TRL Version:** This pipeline uses `trl>=0.8.6` as defined in `requirements.txt`. The PPO training script (`scripts/train_ppo.py`) is compatible with this version.
*   **Dataset Acquisition:** The example commands below use placeholder dataset names (`UltraChat.jsonl`, `OpenHermes_pairs.jsonl`). You will need to procure or create your own datasets in the expected formats. The `scripts/prepare_data.py` script processes data from `data/` into `data/processed/`. Ensure your raw data (e.g., `UltraChat.jsonl`, `OpenHermes_pairs.jsonl`) is placed in the `data/` directory before running `prepare_data.py`.
*   **Model Quality with Quick Start:** The quick start commands use minimal training steps (`--max_steps 2` or `--steps 2`). This is for rapidly testing the pipeline functionality and will **not** produce a high-quality, usable chat model. Expect repetitive or nonsensical output from the web demo if using these minimal settings. For a functional model, significantly more training data and steps are required.
*   **Base Model:** The SFT script defaults to `tiny_llama` if no `--model` argument is provided. You can specify other Hugging Face model IDs or local paths.

```bash
git clone https://github.com/your-handle/open_rlhf_pipeline.git # Replace with the actual repository URL
cd open_rlhf_pipeline

# Create a Python virtual environment (recommended)
# python -m venv .venv
# source .venv/bin/activate

pip install -r requirements.txt

# 1 – Prepare Data
# Ensure your raw instruction (e.g., UltraChat.jsonl) and preference (e.g., OpenHermes_pairs.jsonl)
# data are in the `data/` directory.
# The script will output processed files to `data/processed/`.
python scripts/prepare_data.py --instruct_data data/UltraChat.jsonl \
                               --preference_data data/OpenHermes_pairs.jsonl

# 2 – Supervised Fine-Tuning (SFT)
# Using a small model for demonstration. Replace with your desired base model.
python scripts/train_sft.py --model sshleifer/tiny-gpt2 --output_dir models/sft --max_steps 2 # Example uses tiny-gpt2

# 3 – Reward Model (RM) Training
python scripts/train_reward.py --model models/sft --output_dir models/rm --max_steps 2

# 4 – Policy Optimisation
# Option A: Proximal Policy Optimization (PPO)
python scripts/train_ppo.py --policy models/sft --reward_model_path models/rm --output_dir models/ppo --steps 2

# Option B: Direct Preference Optimization (DPO)
# python scripts/train_dpo.py --policy models/sft --output_dir models/dpo --steps 2

# 5 – Evaluation (Example)
# Ensure the evaluation scripts are adapted for your specific benchmark datasets.
# python eval/run_eval.py --model models/ppo --bench all

# 6 – Safety Red-Teaming (Example)
# Ensure the safety scripts are adapted for your specific needs.
# python eval/run_safety.py --model models/ppo

# 7 – Chat Demo
python web_demo/app.py --model models/ppo
```

## Current Status & Areas for Development

This pipeline provides a solid foundation for RLHF. Here's a summary of what's implemented and what could be enhanced:

**Implemented Features:**
*   **Full RLHF Workflow:** Scripts for data preparation, SFT, Reward Modeling, PPO, and DPO.
*   **Evaluation & Safety Stubs:** Scripts `run_eval.py` and `run_safety.py` are present as starting points.
*   **Interpretability Tools:** Includes `attention_viz.py` and `logit_lens.py`.
*   **Custom Kernel Example:** A `fused_rmsnorm.py` kernel with a benchmark script.
*   **Web Demo:** A Gradio-based chat interface.
*   **Configuration Management:** YAML files for hardware presets.
*   **Dockerization:** `Dockerfile` for containerized environment.
*   **Dependency Management:** `requirements.txt` specifying `trl>=0.8.6`.

**Areas for Contribution & Future Work:**
*   **TRL Version Compatibility:** While currently targeting `trl>=0.8.6`, thoroughly test all scripts, especially `train_ppo.py`, with the latest TRL versions to ensure continued compatibility and leverage new features.
*   **Dataset Handling & Examples:**
    *   Provide example small datasets (if licensing permits) or clear download/formatting scripts for common open-source RLHF datasets.
    *   Enhance `scripts/prepare_data.py` for more robust handling of diverse data formats.
*   **`scaling_laws.ipynb`:** Develop the content of this notebook to provide actual experiments and insights into scaling laws. Currently, it's a placeholder.
*   **Testing Framework:** Populate the `tests/` directory with unit and integration tests for all pipeline components to ensure robustness and reliability.
*   **Evaluation Suite:**
    *   Flesh out `eval/run_eval.py` with support for standard NLP benchmarks (e.g., HELM, MT-Bench, AlpacaEval).
    *   Expand `eval/run_safety.py` with more comprehensive safety evaluation protocols and metrics.
*   **Model Checkpoints & Examples:** Provide (if feasible) example SFT, RM, and PPO/DPO model checkpoints trained on a small, open dataset to facilitate easier experimentation.
*   **Documentation & Tutorials:**
    *   Add more detailed documentation for each script and module.
    *   Create tutorials for advanced use cases (e.g., multi-GPU training, customizing models).
*   **Advanced Kernel Integration:** Explore and integrate more custom kernels for performance optimization.
*   **Configuration Expansion:** Add more pre-defined configurations for various cloud platforms or hardware.
*   **CI/CD Pipeline:** Enhance the `.github/workflows/ci.yml` for automated testing and potentially building Docker images.

## Contributing

I welcome contributions to the Open RLHF Pipeline! If you're interested in helping, please feel free to:
*   Report bugs or issues.
*   Suggest new features or enhancements.
*   Submit pull requests for bug fixes or new functionality.

Please see the (soon to be created) `CONTRIBUTING.md` for more detailed guidelines.

## License
Apache-2.0
