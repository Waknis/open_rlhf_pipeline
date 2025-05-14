"""Simple RMSNorm implementation and benchmark utility."""

import time

import torch
import torch.nn as nn


class SimpleRMSNorm(nn.Module):
    """A simple RMSNorm implementation as a PyTorch nn.Module."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size (int): The size of the hidden dimension.
            eps (float, optional): A small value to prevent division by zero.
                                   Defaults to 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Learnable weight (gamma)
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, expected shape (..., hidden_size).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input tensor last dimension {x.shape[-1]} does not match "
                f"hidden_size {self.hidden_size}"
            )

        # Standard RMSNorm calculation:
        # variance = x.pow(2).mean(-1, keepdim=True)
        # x_normalized = x * torch.rsqrt(variance + self.eps)

        # Alternative calculation (often seen, mathematically equivalent for large hidden_size w.r.t eps):
        # norm_x_sq = torch.mean(x * x, dim=-1, keepdim=True)
        # x_normalized = x * torch.rsqrt(norm_x_sq + self.eps)

        # Using the definition from Llama papers / torch.nn.RMSNorm for clarity:
        # input_dtype = x.dtype
        # x = x.to(torch.float32) # Calculate in float32 for stability
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normalized = x * torch.rsqrt(variance + self.eps)
        # x_normalized = x_normalized.to(input_dtype)

        return self.weight * x_normalized

    def __repr__(self):
        return f"SimpleRMSNorm(hidden_size={self.hidden_size}, eps={self.eps})"


def benchmark_rmsnorm(
    norm_layer: nn.Module,
    input_tensor: torch.Tensor,
    num_iters: int = 100,
    warmup_iters: int = 10,
    layer_name: str = "RMSNormLayer",
) -> None:
    """Benchmarks a given RMSNorm layer.

    Args:
        norm_layer (nn.Module): The RMSNorm layer instance to benchmark.
        input_tensor (torch.Tensor): A sample input tensor.
        num_iters (int, optional): Number of iterations for timing. Defaults to 100.
        warmup_iters (int, optional): Number of warmup iterations. Defaults to 10.
        layer_name (str, optional): Name of the layer for printing. Defaults to "RMSNormLayer".
    """
    device = input_tensor.device
    norm_layer.to(device)
    norm_layer.eval()  # Ensure no training-specific behavior

    # Warmup iterations
    for _ in range(warmup_iters):
        _ = norm_layer(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()  # Wait for all kernels to complete before timing

    start_time = time.perf_counter()
    for _ in range(num_iters):
        _ = norm_layer(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / num_iters) * 1000
    print(f"Benchmark for {layer_name} on {device}:")
    print(f"  Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"  Average time per iteration over {num_iters} runs: {avg_time_ms:.3f} ms")


# This main block is for self-testing or direct execution of this file.
# The primary benchmark script will be bench_rmsnorm.py.
if __name__ == "__main__":
    print("Running SimpleRMSNorm module example & benchmark utility test...")
    hidden_dim_test = 1024  # A more typical hidden dim
    batch_size_test = 4
    seq_len_test = 256

    # Test on CPU
    print("\n--- CPU Benchmarks ---")
    cpu_device = torch.device("cpu")
    sample_input_cpu = torch.randn(
        batch_size_test,
        seq_len_test,
        hidden_dim_test,
        device=cpu_device,
        dtype=torch.float32,
    )

    custom_rmsnorm_cpu = SimpleRMSNorm(hidden_dim_test)
    # Check if torch.nn.RMSNorm is available (PyTorch 1.11+)
    try:
        torch_rmsnorm_cpu = nn.RMSNorm(hidden_dim_test, elementwise_affine=True)
        # Copy weights for fair comparison if needed, though not critical for speed benchmark
        # with torch.no_grad():
        #    torch_rmsnorm_cpu.weight.copy_(custom_rmsnorm_cpu.weight)
    except AttributeError:
        print(
            "torch.nn.RMSNorm not available in this PyTorch version. Skipping its CPU benchmark."
        )
        torch_rmsnorm_cpu = None

    benchmark_rmsnorm(
        custom_rmsnorm_cpu, sample_input_cpu, layer_name="Custom SimpleRMSNorm (CPU)"
    )
    if torch_rmsnorm_cpu:
        benchmark_rmsnorm(
            torch_rmsnorm_cpu, sample_input_cpu, layer_name="torch.nn.RMSNorm (CPU)"
        )

    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n--- CUDA Benchmarks ---")
        cuda_device = torch.device("cuda")
        # Use float16 or bfloat16 on CUDA for more realistic GPU benchmarks if desired
        # For now, stick to float32 for direct comparison, or match typical usage
        sample_input_cuda = sample_input_cpu.to(cuda_device).to(
            torch.float16
        )  # Example with float16
        print(f"CUDA benchmark input dtype: {sample_input_cuda.dtype}")

        custom_rmsnorm_cuda = (
            SimpleRMSNorm(hidden_dim_test).to(cuda_device).to(torch.float16)
        )
        if torch_rmsnorm_cpu:  # If available, create CUDA version
            try:
                torch_rmsnorm_cuda = (
                    nn.RMSNorm(hidden_dim_test, elementwise_affine=True)
                    .to(cuda_device)
                    .to(torch.float16)
                )
                # with torch.no_grad():
                #    torch_rmsnorm_cuda.weight.copy_(custom_rmsnorm_cuda.weight)
            except AttributeError:
                torch_rmsnorm_cuda = (
                    None  # Should not happen if CPU version was available
                )
            except (
                Exception
            ) as e:  # Catch other potential errors e.g. dtype issues on some GPUs
                print(
                    f"Could not initialize torch.nn.RMSNorm on CUDA with specified dtype: {e}"
                )
                torch_rmsnorm_cuda = None
        else:
            torch_rmsnorm_cuda = None

        benchmark_rmsnorm(
            custom_rmsnorm_cuda,
            sample_input_cuda,
            layer_name="Custom SimpleRMSNorm (CUDA)",
        )
        if torch_rmsnorm_cuda:
            benchmark_rmsnorm(
                torch_rmsnorm_cuda,
                sample_input_cuda,
                layer_name="torch.nn.RMSNorm (CUDA)",
            )
    else:
        print("\nCUDA not available, skipping CUDA benchmarks.")
