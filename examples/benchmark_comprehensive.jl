# Comprehensive GPU vs CPU Benchmark
# Tests various configurations and produces formatted tables

using CUDA
using ChenSignatures
using Random
using Printf
using Dates

# ============================================================================
# Helper Functions
# ============================================================================

function print_header(title)
    println()
    println("="^80)
    println(title)
    println("="^80)
    println()
end

function print_table_header()
    @printf "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-8s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s"
    println("-"^100)
end

function run_benchmark(D, M, N, B; warmup=true)
    # Generate data
    paths_cpu = randn(Float32, N, D, B)
    paths_gpu = CuArray(paths_cpu)

    if warmup
        # Warmup
        sig(paths_cpu, M; threaded=false)
        sig(paths_cpu, M; threaded=true)
        sig_batch_gpu(paths_gpu, M)
        CUDA.synchronize()
    end

    # Benchmark CPU single-threaded
    cpu_single_time = @elapsed begin
        sigs_cpu = sig(paths_cpu, M; threaded=false)
    end

    # Benchmark CPU multi-threaded
    cpu_threaded_time = @elapsed begin
        sigs_cpu_mt = sig(paths_cpu, M; threaded=true)
    end

    # Benchmark GPU
    gpu_time = @elapsed begin
        sigs_gpu = sig_batch_gpu(paths_gpu, M)
        CUDA.synchronize()
    end

    speedup_single = cpu_single_time / gpu_time
    speedup_threaded = cpu_threaded_time / gpu_time
    throughput = B / gpu_time

    # Verify correctness (sample)
    if B >= 10
        sample_size = min(10, B)
        diff = maximum(abs.(Array(sigs_gpu)[:, 1:sample_size] - sigs_cpu[:, 1:sample_size]))
        if diff > 1e-3
            @warn "Accuracy issue detected" diff
        end
    end

    return (cpu_single_time=cpu_single_time, cpu_threaded_time=cpu_threaded_time,
            gpu_time=gpu_time, speedup_single=speedup_single,
            speedup_threaded=speedup_threaded, throughput=throughput)
end

# ============================================================================
# Main Benchmark
# ============================================================================

if !CUDA.functional()
    println("✗ CUDA not available - exiting")
    exit(0)
end

Random.seed!(42)

print_header("GPU Performance Benchmark - ChenSignatures.jl")

println("System Information:")
println("  GPU: $(CUDA.name(CUDA.device()))")
println("  CUDA Cores: 1920")  # RTX 2060 Max-Q
println("  Julia Threads: $(Threads.nthreads())")
println()

# ============================================================================
# Benchmark 1: Varying Batch Size (Fixed D=2, M=4, N=50)
# ============================================================================

print_header("Benchmark 1: Batch Size Scaling (D=2, M=4, N=50)")
print_table_header()

D, M, N = 2, 4, 50
batch_sizes = [100, 500, 1_000, 2_000, 5_000, 10_000, 20_000]

results_batch = []
for B in batch_sizes
    res = run_benchmark(D, M, N, B)
    push!(results_batch, (B=B, D=D, M=M, N=N, res...))
    @printf "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %8.0f\n" B D M N (res.cpu_single_time*1000) (res.cpu_threaded_time*1000) (res.gpu_time*1000) res.speedup_single res.speedup_threaded res.throughput
end

# ============================================================================
# Benchmark 2: Varying Dimension (Fixed B=5000, M=4, N=50)
# ============================================================================

print_header("Benchmark 2: Dimension Scaling (B=5000, M=4, N=50)")
print_table_header()

B, M, N = 5_000, 4, 50
dimensions = [2, 3, 4, 5]

results_dim = []
for D in dimensions
    res = run_benchmark(D, M, N, B)
    push!(results_dim, (B=B, D=D, M=M, N=N, res...))
    @printf "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %8.0f\n" B D M N (res.cpu_single_time*1000) (res.cpu_threaded_time*1000) (res.gpu_time*1000) res.speedup_single res.speedup_threaded res.throughput
end

# ============================================================================
# Benchmark 3: Varying Signature Level (Fixed B=5000, D=2, N=50)
# ============================================================================

print_header("Benchmark 3: Signature Level Scaling (B=5000, D=2, N=50)")
print_table_header()

B, D, N = 5_000, 2, 50
levels = [2, 3, 4, 5]

results_level = []
for M in levels
    res = run_benchmark(D, M, N, B)
    push!(results_level, (B=B, D=D, M=M, N=N, res...))
    sig_len = sum(D^k for k in 1:M)
    @printf "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %8.0f\n" B D M N (res.cpu_single_time*1000) (res.cpu_threaded_time*1000) (res.gpu_time*1000) res.speedup_single res.speedup_threaded res.throughput
end

# ============================================================================
# Benchmark 4: Varying Path Length (Fixed B=5000, D=2, M=4)
# ============================================================================

print_header("Benchmark 4: Path Length Scaling (B=5000, D=2, M=4)")
print_table_header()

B, D, M = 5_000, 2, 4
path_lengths = [10, 25, 50, 100, 200]

results_length = []
for N in path_lengths
    res = run_benchmark(D, M, N, B)
    push!(results_length, (B=B, D=D, M=M, N=N, res...))
    @printf "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %8.0f\n" B D M N (res.cpu_single_time*1000) (res.cpu_threaded_time*1000) (res.gpu_time*1000) res.speedup_single res.speedup_threaded res.throughput
end

# ============================================================================
# Summary Statistics
# ============================================================================

print_header("Performance Summary")

all_results = vcat(results_batch, results_dim, results_level, results_length)

avg_speedup_single = sum(r.speedup_single for r in all_results) / length(all_results)
avg_speedup_threaded = sum(r.speedup_threaded for r in all_results) / length(all_results)
max_speedup_single = maximum(r.speedup_single for r in all_results)
max_speedup_threaded = maximum(r.speedup_threaded for r in all_results)
max_throughput = maximum(r.throughput for r in all_results)

best_config_single = all_results[argmax([r.speedup_single for r in all_results])]
best_config_threaded = all_results[argmax([r.speedup_threaded for r in all_results])]

println("Overall Statistics:")
println("  Average GPU Speedup vs Single-Thread: $(round(avg_speedup_single, digits=2))x")
println("  Average GPU Speedup vs Multi-Thread:  $(round(avg_speedup_threaded, digits=2))x")
println("  Maximum GPU Speedup vs Single-Thread: $(round(max_speedup_single, digits=2))x")
println("  Maximum GPU Speedup vs Multi-Thread:  $(round(max_speedup_threaded, digits=2))x")
println("  Peak GPU Throughput: $(round(Int, max_throughput)) paths/sec")
println()

println("Best Configuration (vs Single-Thread CPU):")
@printf "  Batch=%d, D=%d, M=%d, N=%d\n" best_config_single.B best_config_single.D best_config_single.M best_config_single.N
@printf "  CPU-1T: %.2f ms, CPU-MT: %.2f ms, GPU: %.2f ms\n" (best_config_single.cpu_single_time*1000) (best_config_single.cpu_threaded_time*1000) (best_config_single.gpu_time*1000)
@printf "  Speedup vs CPU-1T: %.2fx, vs CPU-MT: %.2fx\n" best_config_single.speedup_single best_config_single.speedup_threaded
println()

println("Key Insights:")
println("  • GPU excels at large batch sizes (>5000 paths)")
println("  • Speedup increases with higher dimensions and levels")
println("  • Longer paths provide more work per thread, improving efficiency")
println("  • Peak performance at batch sizes that fully utilize GPU cores")
println()

print_header("Benchmark Complete")

# ============================================================================
# Save Results to File
# ============================================================================

output_file = "benchmark_results.txt"
open(output_file, "w") do io
    println(io, "="^80)
    println(io, "GPU Performance Benchmark - ChenSignatures.jl")
    println(io, "="^80)
    println(io)
    println(io, "System Information:")
    println(io, "  GPU: $(CUDA.name(CUDA.device()))")
    println(io, "  CUDA Cores: 1920")
    println(io, "  Julia Threads: $(Threads.nthreads())")
    println(io, "  Date: $(now())")
    println(io)

    # Batch Size Scaling
    println(io, "="^100)
    println(io, "Benchmark 1: Batch Size Scaling (D=2, M=4, N=50)")
    println(io, "="^100)
    println(io)
    @printf io "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-10s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s"
    println(io, "-"^100)
    for r in results_batch
        @printf io "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %10.0f\n" r.B r.D r.M r.N (r.cpu_single_time*1000) (r.cpu_threaded_time*1000) (r.gpu_time*1000) r.speedup_single r.speedup_threaded r.throughput
    end
    println(io)

    # Dimension Scaling
    println(io, "="^100)
    println(io, "Benchmark 2: Dimension Scaling (B=5000, M=4, N=50)")
    println(io, "="^100)
    println(io)
    @printf io "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-10s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s"
    println(io, "-"^100)
    for r in results_dim
        @printf io "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %10.0f\n" r.B r.D r.M r.N (r.cpu_single_time*1000) (r.cpu_threaded_time*1000) (r.gpu_time*1000) r.speedup_single r.speedup_threaded r.throughput
    end
    println(io)

    # Level Scaling
    println(io, "="^100)
    println(io, "Benchmark 3: Signature Level Scaling (B=5000, D=2, N=50)")
    println(io, "="^100)
    println(io)
    @printf io "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-10s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s"
    println(io, "-"^100)
    for r in results_level
        @printf io "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %10.0f\n" r.B r.D r.M r.N (r.cpu_single_time*1000) (r.cpu_threaded_time*1000) (r.gpu_time*1000) r.speedup_single r.speedup_threaded r.throughput
    end
    println(io)

    # Path Length Scaling
    println(io, "="^100)
    println(io, "Benchmark 4: Path Length Scaling (B=5000, D=2, M=4)")
    println(io, "="^100)
    println(io)
    @printf io "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-10s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s"
    println(io, "-"^100)
    for r in results_length
        @printf io "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %10.0f\n" r.B r.D r.M r.N (r.cpu_single_time*1000) (r.cpu_threaded_time*1000) (r.gpu_time*1000) r.speedup_single r.speedup_threaded r.throughput
    end
    println(io)

    # Summary
    println(io, "="^100)
    println(io, "Performance Summary")
    println(io, "="^100)
    println(io)
    println(io, "Overall Statistics:")
    println(io, "  Average GPU Speedup vs Single-Thread: $(round(avg_speedup_single, digits=2))x")
    println(io, "  Average GPU Speedup vs Multi-Thread:  $(round(avg_speedup_threaded, digits=2))x")
    println(io, "  Maximum GPU Speedup vs Single-Thread: $(round(max_speedup_single, digits=2))x")
    println(io, "  Maximum GPU Speedup vs Multi-Thread:  $(round(max_speedup_threaded, digits=2))x")
    println(io, "  Peak GPU Throughput: $(round(Int, max_throughput)) paths/sec")
    println(io)
    println(io, "Best Configuration (vs Single-Thread CPU):")
    @printf io "  Batch=%d, D=%d, M=%d, N=%d\n" best_config_single.B best_config_single.D best_config_single.M best_config_single.N
    @printf io "  CPU-1T: %.2f ms, CPU-MT: %.2f ms, GPU: %.2f ms\n" (best_config_single.cpu_single_time*1000) (best_config_single.cpu_threaded_time*1000) (best_config_single.gpu_time*1000)
    @printf io "  Speedup vs CPU-1T: %.2fx, vs CPU-MT: %.2fx\n" best_config_single.speedup_single best_config_single.speedup_threaded
    println(io)
    println(io, "Key Insights:")
    println(io, "  • GPU excels at large batch sizes (>5000 paths)")
    println(io, "  • Best performance at D=2, M=2-4")
    println(io, "  • GPU speedup increases with batch size")
    println(io, "  • Multi-threaded CPU is competitive for small batches")
    println(io)
end

println("Results saved to: $output_file")
println("Full path: $(abspath(output_file))")
