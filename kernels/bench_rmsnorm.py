"""Benchmark script comparing custom SimpleRMSNorm with torch.nn.RMSNorm."""

import argparse  # For potential future CLI args, though not strictly needed by reqs

import torch
import torch.nn as nn

# Attempt to import from the local kernels package
try:
    from .fused_rmsnorm import SimpleRMSNorm, benchmark_rmsnorm
except ImportError:
    # Fallback for running script directly from kernels/ directory or if package structure isn't set up
    # This assumes fused_rmsnorm.py is in the same directory
    try:
        from fused_rmsnorm import SimpleRMSNorm, benchmark_rmsnorm
    except ImportError as e:
        print(f"Error importing RMSNorm modules: {e}")
        print(
            "Please ensure kernels/fused_rmsnorm.py is accessible and defines SimpleRMSNorm and benchmark_rmsnorm."
        )
        SimpleRMSNorm = None  # type: ignore
        benchmark_rmsnorm = None  # type: ignore


def main() -> None:
    """Main function to run RMSNorm benchmarks."""
    if SimpleRMSNorm is None or benchmark_rmsnorm is None:
        print("RMSNorm components not imported. Exiting benchmark.")
        return

    print("Starting RMSNorm benchmark comparison...")

    # Define benchmark parameters
    # Using slightly smaller dimensions for quick CPU test as per user request for CPU-only Mac runs
    # Larger dimensions would be more typical for GPU benchmarks.
    batch_size = 4
    # seq_len = 1024
    # hidden_dim = 2048
    seq_len = 256  # Smaller for quicker CPU run
    hidden_dim = 1024  # Smaller for quicker CPU run
    num_iters = 50  # Fewer iterations for quicker run
    warmup_iters = 5

    print(
        f"Benchmark Parameters: Batch={batch_size}, SeqLen={seq_len}, HiddenDim={hidden_dim}"
    )
    print(f"Iterations: {num_iters} (Warmup: {warmup_iters})")

    # --- CPU Benchmark ---
    print("\n--- CPU Benchmarks (float32) ---")
    cpu_device = torch.device("cpu")
    # Ensure input is float32 for CPU baseline as torch.nn.RMSNorm might default to this
    sample_input_cpu = torch.randn(
        batch_size, seq_len, hidden_dim, device=cpu_device, dtype=torch.float32
    )

    # Custom RMSNorm
    custom_rmsnorm_cpu = SimpleRMSNorm(hidden_dim).to(cpu_device)
    benchmark_rmsnorm(
        custom_rmsnorm_cpu,
        sample_input_cpu,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        layer_name="Custom SimpleRMSNorm (CPU)",
    )

    # PyTorch's nn.RMSNorm
    try:
        torch_rmsnorm_cpu = nn.RMSNorm(hidden_dim, elementwise_affine=True).to(
            cpu_device
        )
        # For a truly fair comparison of the operation itself, elementwise_affine should match.
        # SimpleRMSNorm has it by default (self.weight). nn.RMSNorm needs it specified.
        benchmark_rmsnorm(
            torch_rmsnorm_cpu,
            sample_input_cpu,
            num_iters=num_iters,
            warmup_iters=warmup_iters,
            layer_name="torch.nn.RMSNorm (CPU)",
        )
    except AttributeError:
        print(
            "torch.nn.RMSNorm not available in this PyTorch version. Skipping its CPU benchmark."
        )
    except Exception as e:
        print(
            f"Error setting up torch.nn.RMSNorm on CPU: {e}. Skipping its CPU benchmark."
        )

    # --- CUDA Benchmark (if available) ---
    if torch.cuda.is_available():
        print("\n--- CUDA Benchmarks --- ")
        cuda_device = torch.device("cuda")

        # Try with float16 for a common GPU scenario
        # Note: bfloat16 could also be used if supported: torch.cuda.is_bf16_supported()
        cuda_dtype = torch.float16
        print(f"Using dtype: {cuda_dtype} for CUDA benchmarks")

        sample_input_cuda = torch.randn(
            batch_size, seq_len, hidden_dim, device=cuda_device, dtype=cuda_dtype
        )

        # Custom RMSNorm on CUDA
        try:
            custom_rmsnorm_cuda = (
                SimpleRMSNorm(hidden_dim).to(cuda_device).to(cuda_dtype)
            )
            benchmark_rmsnorm(
                custom_rmsnorm_cuda,
                sample_input_cuda,
                num_iters=num_iters,
                warmup_iters=warmup_iters,
                layer_name=f"Custom SimpleRMSNorm (CUDA, {cuda_dtype})",
            )
        except Exception as e:
            print(
                f"Error setting up Custom SimpleRMSNorm on CUDA with {cuda_dtype}: {e}. Skipping."
            )

        # PyTorch's nn.RMSNorm on CUDA
        try:
            torch_rmsnorm_cuda = (
                nn.RMSNorm(hidden_dim, elementwise_affine=True)
                .to(cuda_device)
                .to(cuda_dtype)
            )
            benchmark_rmsnorm(
                torch_rmsnorm_cuda,
                sample_input_cuda,
                num_iters=num_iters,
                warmup_iters=warmup_iters,
                layer_name=f"torch.nn.RMSNorm (CUDA, {cuda_dtype})",
            )
        except AttributeError:
            print(
                "torch.nn.RMSNorm not available in this PyTorch version. Skipping its CUDA benchmark."
            )
        except Exception as e:
            # This can happen if the specific GPU doesn't support the dtype well with nn.RMSNorm
            print(
                f"Error setting up torch.nn.RMSNorm on CUDA with {cuda_dtype}: {e}. Skipping."
            )
    else:
        print("\nCUDA not available, skipping CUDA benchmarks.")

    print("\nRMSNorm benchmark comparison complete.")


if __name__ == "__main__":
    main()
