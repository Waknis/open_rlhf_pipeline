"""Display top-K token predictions from each layer's hidden states (Logit Lens)."""

import argparse
import pathlib  # Not strictly needed for CLI args if not using Path type, but good practice

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    """Main function to run logit lens."""
    ap = argparse.ArgumentParser(
        description="Run Logit Lens on a model for given text."
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
        default="A quick brown fox",
        help="Input text to apply logit lens to.",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top token predictions to display.",
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="Load model in bfloat16 if CUDA is available.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float32
    if args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        print(f"Using bfloat16 for model {args.model_id}.")
    else:
        print(f"Using float32 for model {args.model_id}.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        # Crucial: output_hidden_states=True to get intermediate layer representations
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, output_hidden_states=True, torch_dtype=model_dtype
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model {args.model_id}: {e}")
        return

    inputs = tokenizer(args.text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_dim)

    if not hidden_states:
        print("Error: Hidden states not found. Ensure `output_hidden_states=True`.")
        return

    # The first item in hidden_states is the embedding output, then one for each layer
    # For lm_head, we usually apply it to the output of each transformer layer.
    # hidden_states[0] is embeddings, hidden_states[1] is L0 output, ..., hidden_states[N] is L(N-1) output

    print(f'Logit Lens for text: "{args.text}"')
    print(f"Model: {args.model_id}\n")

    # Iterate through each layer's hidden states (including embedding layer output)
    # Layer 0 is embedding output, Layer 1 is output of Transformer Block 0, etc.
    for layer_idx, layer_hidden_state in enumerate(hidden_states):
        # layer_hidden_state shape: (batch_size, seq_len, hidden_size)
        # Project to vocabulary space using the model's lm_head
        # Some models (like non-decoder-only) might not have a direct lm_head or require different handling.
        # Assuming standard CausalLM with an lm_head.
        if not hasattr(model, "lm_head") or model.lm_head is None:
            print(
                f"Warning: Model {args.model_id} does not have an lm_head or it is None. Skipping logit projection."
            )
            # For models without a separate lm_head (e.g. if tied embeddings are handled differently),
            # one might need to use model.get_output_embeddings().weight (the vocab matrix)
            # and do manual matrix multiplication: layer_hidden_state @ model.get_output_embeddings().weight.t()
            # However, `model.lm_head(layer_hidden_state)` is the standard way for most causal LMs.
            # If lm_head is an Identity (e.g. if output_embedding itself is the logit layer), this will fail.
            # For robustness, we might need to check model config or type.
            # For now, proceed assuming a typical CausalLM structure.
            try:
                # Attempt to use the output embedding matrix directly if lm_head is problematic
                # This is a common case when lm_head is just an identity layer and weights are tied.
                output_embedding_matrix = model.get_output_embeddings().weight
                layer_logits = torch.matmul(
                    layer_hidden_state, output_embedding_matrix.t()
                )
            except AttributeError:
                print(
                    f"Could not get lm_head or output_embeddings for layer {layer_idx}. Skipping layer."
                )
                continue
        else:
            layer_logits = model.lm_head(
                layer_hidden_state
            )  # Shape: (batch_size, seq_len, vocab_size)

        print(f"--- Layer {layer_idx} ---")

        for token_idx in range(layer_logits.size(1)):  # Iterate over sequence length
            current_token_str = input_tokens[token_idx]

            # Get logits for the current token position: (vocab_size)
            position_logits = layer_logits[0, token_idx, :]

            # Get top K predictions
            top_k_probs, top_k_indices = torch.topk(
                torch.softmax(position_logits, dim=-1), args.top_k
            )
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

            predictions_str = ", ".join(
                [
                    f"'{tok}' ({prob:.3f})"
                    for tok, prob in zip(top_k_tokens, top_k_probs)
                ]
            )
            print(
                f"  Position {token_idx} ('{current_token_str}'): Top {args.top_k} -> {predictions_str}"
            )
        print("")  # Newline for readability between layers

    print("✓ Logit lens analysis complete.")


if __name__ == "__main__":
    main()
