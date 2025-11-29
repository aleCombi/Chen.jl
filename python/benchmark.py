import torch
import timeit
import numpy as np
from chen.torch import sig_torch

def benchmark_gradient(N=500, d=5, m=4):
    print("=" * 70)
    print(f"PYTHON GRADIENT BENCHMARK: N={N}, d={d}, m={m}")
    print("=" * 70)

    # Julia uses Float64 by default. PyTorch uses Float32.
    # We must use Float64 to make the benchmark fair and avoid casting overhead.
    torch.set_default_dtype(torch.float64)
    
    # ---------------------------------------------------------
    # 1. Forward only
    # ---------------------------------------------------------
    print("\nForward only:")
    
    path = torch.randn(N, d, requires_grad=False)
    
    # Warmup: Run a few times to settle JIT and caches
    for _ in range(10):
        sig_torch(path, m)
    
    # Benchmarking
    t_fwd = timeit.Timer(lambda: sig_torch(path, m))
    
    # Run multiple repeats, take the minimum (standard benchmarking practice)
    number_loops = 50
    results_fwd = t_fwd.repeat(repeat=7, number=number_loops)
    min_time_fwd = min(results_fwd) / number_loops
    
    print(f"  {min_time_fwd * 1000:.3f} ms")

    # ---------------------------------------------------------
    # 2. Forward + Backward
    # ---------------------------------------------------------
    print("\nForward + Backward (Autograd):")
    
    path_grad = torch.randn(N, d, requires_grad=True)
    
    def run_backward_step():
        # 1. Clear old gradients (crucial! accumulating gradients is slower)
        path_grad.grad = None
        
        # 2. Forward
        res = sig_torch(path_grad, m)
        
        # 3. Loss (sum is simple and differentiable)
        loss = res.sum()
        
        # 4. Backward
        loss.backward()

    # Warmup: Explicitly trigger the backward compilation and allocator setup
    print("  Warming up JIT...", end="", flush=True)
    for _ in range(10):
        run_backward_step()
    print(" Done.")
    
    # Benchmarking
    t_bwd = timeit.Timer(run_backward_step)
    
    # Backward is slower, so we reduce loop count slightly to keep total time reasonable
    number_loops_bwd = 20
    results_bwd = t_bwd.repeat(repeat=7, number=number_loops_bwd)
    min_time_bwd = min(results_bwd) / number_loops_bwd
    
    print(f"  {min_time_bwd * 1000:.3f} ms")
    
    # Sanity check to ensure gradients actually flowed
    assert path_grad.grad is not None
    print(f"  (Sanity check - Grad shape: {tuple(path_grad.grad.shape)})")
    print("=" * 70)

if __name__ == "__main__":
    # Matches your requested N=500
    benchmark_gradient(500, 5, 4)