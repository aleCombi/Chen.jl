# GPU Performance Benchmark - Large Batch
# Test with 10K paths to see GPU advantage

using CUDA
using ChenSignatures
using Random
using Printf

println("="^70)
println("GPU Large Batch Performance Benchmark")
println("="^70)
println()

if !CUDA.functional()
    println("✗ CUDA not available - exiting")
    exit(0)
end

println("GPU: $(CUDA.name(CUDA.device()))")
println("Threads: $(Threads.nthreads())")
println()

# Test parameters
Random.seed!(42)
D = 2      # Dimension
M = 4      # Level
N = 50     # Path length (increased from 20)
B = 10_000 # Batch size - 10K paths!

println("Configuration:")
println("  Dimension (D): $D")
println("  Level (M): $M")
println("  Path length (N): $N")
println("  Batch size (B): $(B)")
println("  Signature length: $(sum(D^k for k in 1:M))")
println()

# Generate test data
println("Generating $(B) random paths...")
paths_cpu = randn(Float32, N, D, B)
println("  Memory: $(round(sizeof(paths_cpu) / 1024^2, digits=2)) MB")
println()

# Transfer to GPU
println("Transferring to GPU...")
paths_gpu = CuArray(paths_cpu)
println("  ✓ Data on GPU")
println()

# ============================================================================
# Benchmark CPU (single-threaded)
# ============================================================================
println("Benchmark 1: CPU Single-Threaded")
println("-"^70)
cpu_single_time = @elapsed begin
    sigs_cpu_single = sig(paths_cpu, M; threaded=false)
end
println("  Time: $(round(cpu_single_time * 1000, digits=2)) ms")
println("  Throughput: $(round(B / cpu_single_time, digits=0)) paths/sec")
println()

# ============================================================================
# Benchmark CPU (multi-threaded)
# ============================================================================
println("Benchmark 2: CPU Multi-Threaded ($(Threads.nthreads()) threads)")
println("-"^70)
cpu_threaded_time = @elapsed begin
    sigs_cpu_threaded = sig(paths_cpu, M; threaded=true)
end
println("  Time: $(round(cpu_threaded_time * 1000, digits=2)) ms")
println("  Throughput: $(round(B / cpu_threaded_time, digits=0)) paths/sec")
println("  Speedup vs single: $(round(cpu_single_time / cpu_threaded_time, digits=2))x")
println()

# ============================================================================
# Benchmark GPU
# ============================================================================
println("Benchmark 3: GPU")
println("-"^70)

# Warmup
println("  Warming up...")
sigs_gpu = sig_batch_gpu(paths_gpu, M)
CUDA.synchronize()

# Actual benchmark
println("  Running benchmark...")
gpu_time = @elapsed begin
    sigs_gpu = sig_batch_gpu(paths_gpu, M)
    CUDA.synchronize()
end
println("  Time: $(round(gpu_time * 1000, digits=2)) ms")
println("  Throughput: $(round(B / gpu_time, digits=0)) paths/sec")
println()

# ============================================================================
# Verify Correctness
# ============================================================================
println("Correctness Check")
println("-"^70)
# Compare first 100 paths
sigs_gpu_cpu = Array(sigs_gpu)[:, 1:100]
sigs_cpu_sample = sigs_cpu_single[:, 1:100]

max_diff = maximum(abs.(sigs_gpu_cpu - sigs_cpu_sample))
rel_error = max_diff / (maximum(abs.(sigs_cpu_sample)) + 1e-8)

println("  Max absolute difference: $(round(max_diff, sigdigits=4))")
println("  Relative error: $(round(rel_error*100, sigdigits=3))%")

if rel_error < 1e-4
    println("  ✓ GPU results match CPU")
else
    println("  ✗ GPU results differ from CPU!")
end
println()

# ============================================================================
# Summary
# ============================================================================
println("="^70)
println("Performance Summary")
println("="^70)
println()

@printf "%-25s %12s %12s\n" "Method" "Time (ms)" "Throughput"
println("-"^70)
@printf "%-25s %12.2f %12.0f\n" "CPU Single-Thread" (cpu_single_time * 1000) (B / cpu_single_time)
@printf "%-25s %12.2f %12.0f\n" "CPU Multi-Thread" (cpu_threaded_time * 1000) (B / cpu_threaded_time)
@printf "%-25s %12.2f %12.0f\n" "GPU" (gpu_time * 1000) (B / gpu_time)
println()

println("Speedup vs CPU (single-threaded): $(round(cpu_single_time / gpu_time, digits=2))x")
println("Speedup vs CPU (multi-threaded):  $(round(cpu_threaded_time / gpu_time, digits=2))x")
println()

# Show GPU utilization estimate
println("GPU Utilization:")
gpu_cores = 1920  # RTX 2060 Max-Q
threads_used = B
util_pct = min(100.0, (threads_used / gpu_cores) * 100)
println("  CUDA cores: $gpu_cores")
println("  Threads launched: $threads_used")
println("  Estimated utilization: $(round(util_pct, digits=1))%")
println()

println("="^70)
