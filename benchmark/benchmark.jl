# benchmark.jl

using Revise
using BenchmarkTools
using StaticArrays
using ChenSignatures
using Printf
using Dates
using DelimitedFiles
using YAML

# -------- config loading --------

function load_config()
    config_path = joinpath(@__DIR__, "benchmark_config.yaml")
    @assert isfile(config_path) "Config file not found: $config_path"

    cfg = YAML.load_file(config_path)

    Ns       = get(cfg, "Ns", [150, 1000, 2000])
    Ds       = get(cfg, "Ds", [2, 6, 7, 8])
    Ms       = get(cfg, "Ms", [4, 6])
    path_str = get(cfg, "path_kind", "linear")
    runs_dir = get(cfg, "runs_dir", "runs")
    repeats  = get(cfg, "repeats", 5)
    operations_raw = get(cfg, "operations", ["signature", "logsignature"])

    path_kind = Symbol(path_str)
    operations = Symbol.(operations_raw)

    return (Ns = Ns, Ds = Ds, Ms = Ms,
            path_kind = path_kind, runs_dir = runs_dir,
            repeats = repeats, operations = operations)
end

# -------- path generators --------

# linear: [t, 2t, 2t, ...]
function make_path_linear_svec(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    [SVector{d,Float64}(ntuple(i -> (i == 1 ? t : 2t), d)) for t in ts]
end

function make_path_linear_matrix(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    path = Matrix{Float64}(undef, N, d)
    path[:, 1] .= ts
    for j in 2:d
        path[:, j] .= 2 .* ts
    end
    return path
end

# sinusoid: [sin(2π·1·t), sin(2π·2·t), ...]
function make_path_sin_svec(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    ω = 2π
    [SVector{d,Float64}(ntuple(i -> sin(ω * i * t), d)) for t in ts]
end

function make_path_sin_matrix(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    ω = 2π
    path = Matrix{Float64}(undef, N, d)
    for j in 1:d
        path[:, j] .= sin.(ω * j .* ts)
    end
    return path
end

function make_path(d::Int, N::Int, kind::Symbol, path_type::Symbol)
    if kind === :linear
        return path_type === :Matrix ? make_path_linear_matrix(d, N) : make_path_linear_svec(d, N)
    elseif kind === :sin
        return path_type === :Matrix ? make_path_sin_matrix(d, N) : make_path_sin_svec(d, N)
    else
        error("Unknown path_kind: $kind (expected :linear or :sin)")
    end
end

# -------- one benchmark case (Julia only) --------

function bench_case(d::Int, m::Int, N::Int, path_kind::Symbol, op::Symbol, path_type::Symbol, repeats::Int)
    path = make_path(d, N, path_kind, path_type)
    tensor_type = ChenSignatures.Tensor{Float64}

    # Determine path type string for output
    path_type_str = path_type === :Matrix ? "Matrix" : "Vector{SVector}"

    # Helper closures for the two operations
    run_sig() = signature_path(tensor_type, path, m)
    run_logsig() = ChenSignatures.log(signature_path(tensor_type, path, m))

    # Select function and warmup
    if op === :signature
        method_name = "signature_path"
        run_sig()
        # Note: We must interpolate local functions with $ for BenchmarkTools
        t_jl = @belapsed $run_sig() evals=1 samples=repeats
        a_jl = @allocated run_sig()
    elseif op === :logsignature
        method_name = "log"
        run_logsig()
        t_jl = @belapsed $run_logsig() evals=1 samples=repeats
        a_jl = @allocated run_logsig()
    else
        error("Unknown operation: $op")
    end

    t_ms      = t_jl * 1000
    alloc_KiB = a_jl / 1024

    return (N = N,
            d = d,
            m = m,
            path_kind = path_kind,
            operation = op,
            language = "julia",
            library = "ChenSignatures.jl",
            method = method_name,
            path_type = path_type_str,
            t_ms = t_ms,
            alloc_KiB = alloc_KiB)
end

# -------- sweep + write grid to file --------

function setup_run_folder(run_type::String, runs_root::String)
    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    run_dir = joinpath(runs_root, "$(run_type)_$(ts)")
    mkpath(run_dir)
    
    println("Created run folder: $run_dir")
    return run_dir, ts
end

function write_metadata(run_dir::String, run_type::String, ts::String, cfg)
    # Write config snapshot
    config_src = joinpath(@__DIR__, "benchmark_config.yaml")
    if isfile(config_src)
        cp(config_src, joinpath(run_dir, "benchmark_config.yaml"); force=true)
    end
    
    # Write resolved config
    config_resolved = joinpath(run_dir, "config_resolved.txt")
    open(config_resolved, "w") do f
        println(f, "Resolved Configuration")
        println(f, "=" ^ 60)
        println(f, "Ns:         $(cfg.Ns)")
        println(f, "Ds:         $(cfg.Ds)")
        println(f, "Ms:         $(cfg.Ms)")
        println(f, "path_kind:  $(cfg.path_kind)")
        println(f, "operations: $(cfg.operations)")
        println(f, "repeats:    $(cfg.repeats)")
    end
    
    # Write metadata
    metadata_file = joinpath(run_dir, "run_metadata.txt")
    open(metadata_file, "w") do f
        println(f, "Run Type: $run_type")
        println(f, "Timestamp: $ts")
        println(f, "Start Time: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    end
    
    return metadata_file
end

function finalize_run_folder(run_dir::String, summary::Dict)
    summary_file = joinpath(run_dir, "SUMMARY.txt")
    open(summary_file, "w") do f
        println(f, "Benchmark Run Summary")
        println(f, "=" ^ 60)
        println(f)
        println(f, "Results:")
        println(f, "-" ^ 60)
        for (key, value) in summary
            println(f, "$key: $value")
        end
        println(f)
        println(f, "End Time: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    end
    
    println("=" ^ 60)
    println("Run completed. Summary written to: $summary_file")
end

function run_bench()
    cfg = load_config()
    Ns, Ds, Ms = cfg.Ns, cfg.Ds, cfg.Ms
    path_kind  = cfg.path_kind
    runs_root  = joinpath(@__DIR__, cfg.runs_dir)
    repeats    = cfg.repeats
    operations = cfg.operations

    # Benchmark both path types
    path_types = [:Matrix, :VectorSVector]

    println("=" ^ 60)
    println("Julia Benchmark Suite")
    println("=" ^ 60)
    println("Configuration:")
    println("  path_kind  = $path_kind")
    println("  Ns         = $(Ns)")
    println("  Ds         = $(Ds)")
    println("  Ms         = $(Ms)")
    println("  operations = $(operations)")
    println("  path_types = $(path_types)")
    println("  repeats    = $repeats")
    println()

    # Check if orchestrator is overriding output location
    custom_csv = get(ENV, "BENCHMARK_OUT_CSV", "")
    if !isempty(custom_csv)
        # Orchestrator mode: write to specified location
        run_dir = dirname(custom_csv)
        file = custom_csv
        println("Orchestrator mode: writing to $file")
    else
        # Standalone mode: create own run folder
        run_dir, ts = setup_run_folder("benchmark_julia", runs_root)
        write_metadata(run_dir, "benchmark_julia", ts, cfg)
        file = joinpath(run_dir, "results.csv")
    end

    results = NamedTuple[]

    for N in Ns, d in Ds, m in Ms, op in operations, ptype in path_types
        push!(results, bench_case(d, m, N, path_kind, op, ptype, repeats))
    end

    # Write results
    header = ["N", "d", "m", "path_kind", "operation", "language", "library", "method", "path_type", "t_ms", "alloc_KiB"]
    data = Array{Any}(undef, length(results) + 1, length(header))
    data[1, :] = header

    for (i, r) in enumerate(results)
        data[i + 1, 1] = r.N
        data[i + 1, 2] = r.d
        data[i + 1, 3] = r.m
        data[i + 1, 4] = String(r.path_kind)
        data[i + 1, 5] = String(r.operation)
        data[i + 1, 6] = r.language
        data[i + 1, 7] = r.library
        data[i + 1, 8] = r.method
        data[i + 1, 9] = r.path_type
        data[i + 1, 10] = r.t_ms
        data[i + 1, 11] = r.alloc_KiB
    end

    writedlm(file, data, ',')

    println("=" ^ 60)
    println("Results written to: $file")
    
    # Only write summary in standalone mode
    if isempty(custom_csv)
        summary = Dict(
            "total_benchmarks" => length(results),
            "path_types" => join(unique([r.path_type for r in results]), ", "),
            "operations" => join(unique([String(r.operation) for r in results]), ", "),
            "output_csv" => basename(file)
        )
        
        finalize_run_folder(run_dir, summary)
    end
    
    return file
end

run_bench()