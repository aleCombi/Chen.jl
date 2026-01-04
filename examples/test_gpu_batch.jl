# Test GPU Batch Processing
# Demonstrates sig_batch_gpu function

using CUDA
using ChenSignatures
using Random

println("Testing GPU Batch Processing\n" * "="^60)

if !CUDA.functional()
    println("✗ CUDA not available - skipping GPU tests")
    exit(0)
end

println("✓ CUDA available: $(CUDA.name(CUDA.device()))")
println()

# Test parameters
Random.seed!(42)
D = 2    # Dimension
M = 4    # Level
N = 20   # Path length
B_small = 10   # Small batch for testing correctness
B_large = 1000  # Large batch for benchmarking

println("Test Configuration:")
println("  Dimension (D): $D")
println("  Level (M): $M")
println("  Path length (N): $N")
println()

# ============================================================================
# Test 1: Correctness with small batch
# ============================================================================
println("Test 1: Correctness Verification")
println("-"^60)

# Generate test data
paths_cpu = randn(Float32, N, D, B_small)

# Compute on CPU (reference)
sigs_cpu = sig(paths_cpu, M)
println("  ✓ CPU computation done ($(size(sigs_cpu)))")

# Move to GPU
paths_gpu = CuArray(paths_cpu)
println("  ✓ Transferred $(B_small) paths to GPU")

# Compute on GPU
CUDA.@allowscalar begin
    sigs_gpu = sig_batch_gpu(paths_gpu, M)
end
println("  ✓ GPU computation done ($(size(sigs_gpu)))")

# Compare results
sigs_gpu_cpu = Array(sigs_gpu)
max_diff = maximum(abs.(sigs_cpu - sigs_gpu_cpu))
rel_error = max_diff / (maximum(abs.(sigs_cpu)) + 1e-8)

println("  ")
println("  Results:")
println("    Max absolute difference: $(round(max_diff, sigdigits=4))")
println("    Relative error: $(round(rel_error*100, sigdigits=3))%")

if rel_error < 1e-4
    println("  ✓ GPU results match CPU (within tolerance)")
else
    println("  ✗ GPU results differ from CPU!")
    exit(1)
end

println()

# ============================================================================
# Test 2: Performance with larger batch
# ============================================================================
println("Test 2: Performance Benchmark")
println("-"^60)

# Generate larger batch
paths_large_cpu = randn(Float32, N, D, B_large)
paths_large_gpu = CuArray(paths_large_cpu)
println("  ✓ Created batch of $B_large paths")

# Benchmark CPU (single-threaded)
println("  ")
println("  CPU (single-threaded):")
cpu_time = @elapsed begin
    sigs_cpu_large = sig(paths_large_cpu, M; threaded=false)
end
println("    Time: $(round(cpu_time * 1000, digits=2)) ms")

# Benchmark CPU (multi-threaded)
println("  ")
println("  CPU ($(Threads.nthreads()) threads):")
cpu_threaded_time = @elapsed begin
    sigs_cpu_threaded = sig(paths_large_cpu, M; threaded=true)
end
println("    Time: $(round(cpu_threaded_time * 1000, digits=2)) ms")
println("    Speedup vs single-thread: $(round(cpu_time / cpu_threaded_time, digits=2))x")

# Benchmark GPU (with warmup)
println("  ")
println("  GPU:")
# Warmup
CUDA.@allowscalar begin
    _ = sig_batch_gpu(paths_large_gpu, M)
end
CUDA.synchronize()

# Actual benchmark
gpu_time = @elapsed begin
    CUDA.@allowscalar begin
        sigs_gpu_large = sig_batch_gpu(paths_large_gpu, M)
    end
    CUDA.synchronize()
end
println("    Time: $(round(gpu_time * 1000, digits=2)) ms")
println("    Speedup vs CPU (single): $(round(cpu_time / gpu_time, digits=2))x")
println("    Speedup vs CPU (threaded): $(round(cpu_threaded_time / gpu_time, digits=2))x")

println()
println("="^60)
println("GPU Batch Processing Tests Complete!")
println()
println("NOTE: Current implementation uses scalar indexing for validation.")
println("      Future optimizations will provide 20-50x speedup with proper")
println("      GPU kernels and batch-level parallelization.")
