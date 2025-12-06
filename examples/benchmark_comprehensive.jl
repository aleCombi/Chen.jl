# Comprehensive GPU vs CPU Benchmark
# Tests various configurations and produces formatted tables
# Configure B/D/M/N via `benchmark_comprehensive_config.toml` (or set BENCH_CONFIG to a custom path).

using CUDA
using ChenSignatures
using Random
using Printf
using Dates
using BenchmarkTools
using Statistics
using TOML
const BENCH_SECONDS = 1.0   # target time budget per benchmarked expression (longer for stability)
const BENCH_SAMPLES = 50    # independent samples to aggregate (use median to avoid outliers)
const BENCH_WARMUP  = 1     # discard this many initial samples when computing the median
const BENCH_EVALS   = 1     # evaluations per sample (keep 1 to avoid extra GPU/CPU work per sample)
const RECOMMENDED_THREADS = max(Threads.nthreads(), Sys.CPU_THREADS)
const MULTITHREAD_ENABLED = Threads.nthreads() > 1
const TARGET_THREADS = get(ENV, "BENCH_THREADS", "")
const DEFAULT_CONFIG_PATH = joinpath(@__DIR__, "benchmark_comprehensive_config.toml")

struct BenchConfig
    sections::Vector{Symbol}
    batch_sizes::Vector{Int}
    batch_dim::Int
    batch_level::Int
    batch_length::Int
    dimensions::Vector{Int}
    dimension_batch::Int
    dimension_level::Int
    dimension_length::Int
    levels::Vector{Int}
    level_batch::Int
    level_dim::Int
    level_length::Int
    path_lengths::Vector{Int}
    length_batch::Int
    length_dim::Int
    length_level::Int
    single_runs::Vector{NamedTuple}
end

# Reuse identical data across sections to avoid variance from different random draws
const _PATH_CACHE = Dict{NTuple{4,Int},NamedTuple{(:paths_cpu, :paths_gpu)}}()

# ============================================================================
# Config loading helpers
# ============================================================================

function parse_int_vec(val, default)
    isa(val, AbstractVector) || return default
    try
        return [Int(x) for x in val]
    catch
        return default
    end
end

function parse_int(val, default)
    try
        return Int(val)
    catch
        return default
    end
end

function parse_sections(val)
    if isa(val, AbstractVector)
        syms = Symbol.(val)
        allowed = [:batch, :dimension, :level, :length, :single]
        return [s for s in syms if s in allowed]
    end
    return [:batch, :dimension, :level, :length, :single]
end

function parse_single_runs(val)
    runs = NamedTuple[]
    if isa(val, AbstractVector)
        for entry in val
            if isa(entry, AbstractDict)
                ok_keys = all(k -> haskey(entry, k), ["B", "D", "M", "N"])
                name = get(entry, "name", "custom")
                if ok_keys
                    push!(runs, (name = String(name),
                                 B = Int(entry["B"]),
                                 D = Int(entry["D"]),
                                 M = Int(entry["M"]),
                                 N = Int(entry["N"])))
                end
            end
        end
    end
    return runs
end

function load_config()
    cfg_path = get(ENV, "BENCH_CONFIG", DEFAULT_CONFIG_PATH)
    if isfile(cfg_path)
        parsed = TOML.parsefile(cfg_path)
        println("Loaded benchmark config: $cfg_path")
    else
        parsed = Dict{String,Any}()
        println("Using built-in defaults (config not found at $cfg_path)")
    end

    sections          = parse_sections(get(parsed, "sections", nothing))

    batch_sizes       = parse_int_vec(get(parsed, "batch_sizes", nothing),
                                      [100, 500, 1_000, 2_000, 5_000, 10_000, 20_000])
    batch_dim         = parse_int(get(parsed, "batch_dim", nothing), 2)
    batch_level       = parse_int(get(parsed, "batch_level", nothing), 4)
    batch_length      = parse_int(get(parsed, "batch_length", nothing), 50)

    dimensions        = parse_int_vec(get(parsed, "dimensions", nothing), [2, 3, 4, 5])
    dimension_batch   = parse_int(get(parsed, "dimension_batch", nothing), 5_000)
    dimension_level   = parse_int(get(parsed, "dimension_level", nothing), 4)
    dimension_length  = parse_int(get(parsed, "dimension_length", nothing), 50)

    levels            = parse_int_vec(get(parsed, "levels", nothing), [2, 3, 4, 5])
    level_batch       = parse_int(get(parsed, "level_batch", nothing), 5_000)
    level_dim         = parse_int(get(parsed, "level_dim", nothing), 2)
    level_length      = parse_int(get(parsed, "level_length", nothing), 50)

    path_lengths      = parse_int_vec(get(parsed, "path_lengths", nothing), [10, 25, 50, 100, 200])
    length_batch      = parse_int(get(parsed, "length_batch", nothing), 5_000)
    length_dim        = parse_int(get(parsed, "length_dim", nothing), 2)
    length_level      = parse_int(get(parsed, "length_level", nothing), 4)

    single_runs       = parse_single_runs(get(parsed, "single_runs", nothing))

    return BenchConfig(sections, batch_sizes, batch_dim, batch_level, batch_length,
                       dimensions, dimension_batch, dimension_level, dimension_length,
                       levels, level_batch, level_dim, level_length,
                       path_lengths, length_batch, length_dim, length_level,
                       single_runs)
end

# ============================================================================#
# Optional self-reexec with a requested thread count (set BENCH_THREADS=N)
if TARGET_THREADS != ""
    requested = tryparse(Int, TARGET_THREADS)
    if requested === nothing || requested < 1
        @warn "Ignoring invalid BENCH_THREADS=$TARGET_THREADS (must be positive integer)"
    elseif Threads.nthreads() != requested
        if get(ENV, "BENCH_ALREADY_REEXEC", "") == "1"
            @warn "Requested BENCH_THREADS=$requested but process already re-execed; continuing with Threads=$(Threads.nthreads())"
        elseif isnothing(Base.julia_cmd())
            @warn "Cannot re-exec Julia automatically; please rerun with JULIA_NUM_THREADS=$requested"
        else
            env = copy(ENV)
            cmd = Base.julia_cmd()
            withenv("JULIA_NUM_THREADS" => string(requested),
                    "BENCH_ALREADY_REEXEC" => "1") do
                run(`$cmd --project=$(Base.active_project()) $(PROGRAM_FILE)`)
            end
            exit(0)
        end
    end
end

# ============================================================================
# Helper Functions
# ============================================================================

@inline function get_paths(D::Int, M::Int, N::Int, B::Int)
    key = (D, M, N, B)
    if haskey(_PATH_CACHE, key)
        return _PATH_CACHE[key]
    end
    paths_cpu = randn(Float32, N, D, B)
    paths_gpu = CuArray(paths_cpu)
    entry = (paths_cpu = paths_cpu, paths_gpu = paths_gpu)
    _PATH_CACHE[key] = entry
    return entry
end

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

@inline function stable_time_ns(trial)
    times = trial.times
    if isempty(times)
        return NaN
    end
    start_idx = min(length(times), BENCH_WARMUP + 1)
    return Statistics.median(@view times[start_idx:end])
end

function run_benchmark(D, M, N, B; warmup=true)
    # Generate or reuse data once (outside timings)
    data = get_paths(D, M, N, B)
    paths_cpu = data.paths_cpu
    paths_gpu = data.paths_gpu

    if warmup
        # Warmup
        sig(paths_cpu, M; threaded=false)
        sig(paths_cpu, M; threaded=true)
        sig_batch_gpu(paths_gpu, M)
        CUDA.synchronize()
    end

    # Sample outputs (not timed) for correctness check
    sigs_cpu = sig(paths_cpu, M; threaded=false)
    sigs_cpu_mt = sig(paths_cpu, M; threaded=true)
    sigs_gpu = sig_batch_gpu(paths_gpu, M)
    CUDA.synchronize()

    # Benchmark CPU single-threaded
    cpu_single_trial = @benchmark sig($paths_cpu, $M; threaded=false) seconds=BENCH_SECONDS samples=BENCH_SAMPLES evals=BENCH_EVALS
    cpu_single_time = stable_time_ns(cpu_single_trial) / 1e9

    # Benchmark CPU multi-threaded (only meaningful if Threads.nthreads() > 1)
    if MULTITHREAD_ENABLED
        cpu_threaded_trial = @benchmark sig($paths_cpu, $M; threaded=true) seconds=BENCH_SECONDS samples=BENCH_SAMPLES evals=BENCH_EVALS
        cpu_threaded_time = stable_time_ns(cpu_threaded_trial) / 1e9
    else
        cpu_threaded_time = cpu_single_time
    end

    # Benchmark GPU (synchronize to include kernel time)
    gpu_trial = @benchmark begin
        sig_batch_gpu($paths_gpu, $M)
        CUDA.synchronize()
    end seconds=BENCH_SECONDS samples=BENCH_SAMPLES evals=BENCH_EVALS
    gpu_time = stable_time_ns(gpu_trial) / 1e9

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
cfg = load_config()

print_header("GPU Performance Benchmark - ChenSignatures.jl")

println("System Information:")
println("  GPU: $(CUDA.name(CUDA.device()))")
println("  CUDA Cores: 1920")  # RTX 2060 Max-Q
println("  Julia Threads: $(Threads.nthreads())")
if !MULTITHREAD_ENABLED && RECOMMENDED_THREADS > 1
    println("  Note: Multithreading disabled (JULIA_NUM_THREADS=1). For meaningful CPU-MT results, rerun with JULIA_NUM_THREADS=$(RECOMMENDED_THREADS).")
end
println()

# ============================================================================
# Benchmark 1: Varying Batch Size (Fixed D, M, N)
# ============================================================================

results_batch = []
if :batch in cfg.sections
    print_header("Benchmark 1: Batch Size Scaling (D=$(cfg.batch_dim), M=$(cfg.batch_level), N=$(cfg.batch_length))")
    print_table_header()

    D, M, N = cfg.batch_dim, cfg.batch_level, cfg.batch_length
    for B in cfg.batch_sizes
        res = run_benchmark(D, M, N, B)
        push!(results_batch, (B=B, D=D, M=M, N=N, res...))
        @printf "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %8.0f\n" B D M N (res.cpu_single_time*1000) (res.cpu_threaded_time*1000) (res.gpu_time*1000) res.speedup_single res.speedup_threaded res.throughput
    end
end

# ============================================================================
# Benchmark 2: Varying Dimension (Fixed B, M, N)
# ============================================================================

results_dim = []
if :dimension in cfg.sections
    print_header("Benchmark 2: Dimension Scaling (B=$(cfg.dimension_batch), M=$(cfg.dimension_level), N=$(cfg.dimension_length))")
    print_table_header()

    B, M, N = cfg.dimension_batch, cfg.dimension_level, cfg.dimension_length
    for D in cfg.dimensions
        res = run_benchmark(D, M, N, B)
        push!(results_dim, (B=B, D=D, M=M, N=N, res...))
        @printf "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %8.0f\n" B D M N (res.cpu_single_time*1000) (res.cpu_threaded_time*1000) (res.gpu_time*1000) res.speedup_single res.speedup_threaded res.throughput
    end
end

# ============================================================================
# Benchmark 3: Varying Signature Level (Fixed B, D, N)
# ============================================================================

results_level = []
if :level in cfg.sections
    print_header("Benchmark 3: Signature Level Scaling (B=$(cfg.level_batch), D=$(cfg.level_dim), N=$(cfg.level_length))")
    print_table_header()

    B, D, N = cfg.level_batch, cfg.level_dim, cfg.level_length
    for M in cfg.levels
        res = run_benchmark(D, M, N, B)
        push!(results_level, (B=B, D=D, M=M, N=N, res...))
        sig_len = sum(D^k for k in 1:M)
        @printf "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %8.0f\n" B D M N (res.cpu_single_time*1000) (res.cpu_threaded_time*1000) (res.gpu_time*1000) res.speedup_single res.speedup_threaded res.throughput
    end
end

# ============================================================================
# Benchmark 4: Varying Path Length (Fixed B, D, M)
# ============================================================================

results_length = []
if :length in cfg.sections
    print_header("Benchmark 4: Path Length Scaling (B=$(cfg.length_batch), D=$(cfg.length_dim), M=$(cfg.length_level))")
    print_table_header()

    B, D, M = cfg.length_batch, cfg.length_dim, cfg.length_level
    for N in cfg.path_lengths
        res = run_benchmark(D, M, N, B)
        push!(results_length, (B=B, D=D, M=M, N=N, res...))
        @printf "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %8.0f\n" B D M N (res.cpu_single_time*1000) (res.cpu_threaded_time*1000) (res.gpu_time*1000) res.speedup_single res.speedup_threaded res.throughput
    end
end

# ============================================================================
# Benchmark 5: Custom Single Runs (explicit tuples)
# ============================================================================

results_single = []
if :single in cfg.sections && !isempty(cfg.single_runs)
    print_header("Benchmark 5: Custom Single Runs")
    print_table_header()
    for run in cfg.single_runs
        B, D, M, N = run.B, run.D, run.M, run.N
        res = run_benchmark(D, M, N, B)
        push!(results_single, (B=B, D=D, M=M, N=N, res..., name=run.name))
        @printf "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %8.0f  (%s)\n" B D M N (res.cpu_single_time*1000) (res.cpu_threaded_time*1000) (res.gpu_time*1000) res.speedup_single res.speedup_threaded res.throughput run.name
    end
end

# ============================================================================
# Summary Statistics
# ============================================================================

print_header("Performance Summary")

all_results = vcat(results_batch, results_dim, results_level, results_length, results_single)

if isempty(all_results)
    println("No benchmarks executed (check config sections in $(DEFAULT_CONFIG_PATH) or BENCH_CONFIG).")
    exit(0)
end

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
    if !isempty(results_batch)
        println(io, "="^100)
        println(io, "Benchmark 1: Batch Size Scaling (D=$(cfg.batch_dim), M=$(cfg.batch_level), N=$(cfg.batch_length))")
        println(io, "="^100)
        println(io)
        @printf io "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-10s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s"
        println(io, "-"^100)
        for r in results_batch
            @printf io "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %10.0f\n" r.B r.D r.M r.N (r.cpu_single_time*1000) (r.cpu_threaded_time*1000) (r.gpu_time*1000) r.speedup_single r.speedup_threaded r.throughput
        end
        println(io)
    end

    # Dimension Scaling
    if !isempty(results_dim)
        println(io, "="^100)
        println(io, "Benchmark 2: Dimension Scaling (B=$(cfg.dimension_batch), M=$(cfg.dimension_level), N=$(cfg.dimension_length))")
        println(io, "="^100)
        println(io)
        @printf io "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-10s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s"
        println(io, "-"^100)
        for r in results_dim
            @printf io "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %10.0f\n" r.B r.D r.M r.N (r.cpu_single_time*1000) (r.cpu_threaded_time*1000) (r.gpu_time*1000) r.speedup_single r.speedup_threaded r.throughput
        end
        println(io)
    end

    # Level Scaling
    if !isempty(results_level)
        println(io, "="^100)
        println(io, "Benchmark 3: Signature Level Scaling (B=$(cfg.level_batch), D=$(cfg.level_dim), N=$(cfg.level_length))")
        println(io, "="^100)
        println(io)
        @printf io "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-10s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s"
        println(io, "-"^100)
        for r in results_level
            @printf io "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %10.0f\n" r.B r.D r.M r.N (r.cpu_single_time*1000) (r.cpu_threaded_time*1000) (r.gpu_time*1000) r.speedup_single r.speedup_threaded r.throughput
        end
        println(io)
    end

    # Path Length Scaling
    if !isempty(results_length)
        println(io, "="^100)
        println(io, "Benchmark 4: Path Length Scaling (B=$(cfg.length_batch), D=$(cfg.length_dim), M=$(cfg.length_level))")
        println(io, "="^100)
        println(io)
        @printf io "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-10s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s"
        println(io, "-"^100)
        for r in results_length
            @printf io "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %10.0f\n" r.B r.D r.M r.N (r.cpu_single_time*1000) (r.cpu_threaded_time*1000) (r.gpu_time*1000) r.speedup_single r.speedup_threaded r.throughput
        end
        println(io)
    end

    # Custom Single Runs
    if !isempty(results_single)
        println(io, "="^100)
        println(io, "Benchmark 5: Custom Single Runs")
        println(io, "="^100)
        println(io)
        @printf io "%-8s %-6s %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s %-10s %-12s\n" "Batch" "Dim" "Level" "Length" "CPU-1T(ms)" "CPU-MT(ms)" "GPU (ms)" "Speed-1T" "Speed-MT" "GPU-path/s" "Name"
        println(io, "-"^110)
        for r in results_single
            @printf io "%-8d %-6d %-6d %-6d | %10.2f %10.2f %10.2f | %8.2fx %8.2fx %10.0f %-12s\n" r.B r.D r.M r.N (r.cpu_single_time*1000) (r.cpu_threaded_time*1000) (r.gpu_time*1000) r.speedup_single r.speedup_threaded r.throughput r.name
        end
        println(io)
    end

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
end

println("Results saved to: $output_file")
println("Full path: $(abspath(output_file))")
