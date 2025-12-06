# GPU Scaling Benchmark
# Test how performance scales with batch size

using CUDA
using ChenSignatures
using Random
using Printf

if !CUDA.functional()
    println("✗ CUDA not available")
    exit(0)
end

println("GPU Scaling Benchmark - $(CUDA.name(CUDA.device()))")
println("="^70)
println()

D, M, N = 2, 4, 50
Random.seed!(42)

batch_sizes = [100, 500, 1_000, 5_000, 10_000, 20_000, 50_000]

println("Configuration: D=$D, M=$M, N=$N")
println()
@printf "%-12s %12s %12s %12s %12s\n" "Batch Size" "GPU (ms)" "CPU (ms)" "Speedup" "Throughput"
println("-"^70)

for B in batch_sizes
    # Generate data
    paths_cpu = randn(Float32, N, D, B)
    paths_gpu = CuArray(paths_cpu)

    # Warmup
    sig_batch_gpu(paths_gpu, M)
    CUDA.synchronize()

    # Benchmark GPU
    gpu_time = @elapsed begin
        sigs_gpu = sig_batch_gpu(paths_gpu, M)
        CUDA.synchronize()
    end

    # Benchmark CPU (multi-threaded)
    cpu_time = @elapsed begin
        sigs_cpu = sig(paths_cpu, M; threaded=true)
    end

    speedup = cpu_time / gpu_time
    throughput = B / gpu_time

    @printf "%-12d %12.2f %12.2f %12.2fx %12.0f\n" B (gpu_time*1000) (cpu_time*1000) speedup throughput
end

println()
println("="^70)
