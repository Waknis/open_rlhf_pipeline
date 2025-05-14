"""Generate and save an attention heatmap for a given model and text."""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    """Main function to generate and save attention heatmap."""
    ap = argparse.ArgumentParser(
        description="Generate attention heatmap for the first layer."
    )
    ap.add_argument(
        "--model_id",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model ID or local path.",
    )
    ap.add_argument(
        "--text",
        type=str,
        default="Hello world",
        help="Input text to visualize attention for.",
    )
    ap.add_argument(
        "--output_path",
        type=pathlib.Path,
        default=pathlib.Path("interpret/heatmap.png"),
        help="Path to save the attention heatmap PNG image.",
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="Load model in bfloat16 if CUDA is available.",
    )
    args = ap.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float32
    if args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        print(f"Using bfloat16 for model {args.model_id}.")
    else:
        print(f"Using float32 for model {args.model_id}.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            output_attentions=True,  # Crucial for getting attention scores
            torch_dtype=model_dtype,
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model {args.model_id}: {e}")
        return

    # Some models might not have pad_token set, though for a short fixed input it might not matter
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(args.text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # attentions is a tuple of tensors, one for each layer.
    # Each tensor shape: (batch_size, num_heads, seq_length, seq_length)
    attentions = outputs.attentions

    if not attentions:
        print(
            "Error: Attentions not found in model output. Ensure `output_attentions=True`."
        )
        return

    # Get attentions for the first layer
    first_layer_attentions = attentions[
        0
    ]  # Shape: (batch_size, num_heads, seq_len, seq_len)

    # Average across heads for simplicity, take first batch item
    # Shape: (seq_len, seq_len)
    avg_first_layer_attentions = first_layer_attentions.mean(dim=1)[0].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_len = avg_first_layer_attentions.shape[0]

    if len(tokens) != seq_len:
        print(
            f"Warning: Token length ({len(tokens)}) and attention matrix dim ({seq_len}) mismatch. Using sliced tokens."
        )
        # This can happen if tokenizer adds special tokens that are not part of attention matrix in some models/configs
        # Or if seq_len from attention is different due to model specifics. Forcing alignment.
        min_len = min(len(tokens), seq_len)
        tokens = tokens[:min_len]
        avg_first_layer_attentions = avg_first_layer_attentions[:min_len, :min_len]
        if not tokens:
            print("Error: No tokens to display after alignment.")
            return

    # Plotting the heatmap
    fig, ax = plt.subplots(
        figsize=(max(6, len(tokens) * 0.8), max(5, len(tokens) * 0.7))
    )
    im = ax.imshow(avg_first_layer_attentions, cmap="viridis")

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            # Ensure value is scalar before formatting
            val = avg_first_layer_attentions[i, j]
            text_color = (
                "w" if val < 0.5 * avg_first_layer_attentions.max() else "black"
            )  # Adjust threshold as needed
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    ax.set_title(f'Average Attention Heatmap (First Layer) for: "{args.text}"')
    fig.tight_layout()

    try:
        plt.savefig(args.output_path)
        print(f"✓ Attention heatmap saved to {args.output_path}")
    except Exception as e:
        print(f"Error saving heatmap: {e}")


if __name__ == "__main__":
    main()
