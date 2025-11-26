# benchmark.jl

using Revise
using BenchmarkTools
using StaticArrays
using Chen
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
function make_path_linear(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    [SVector{d,Float64}(ntuple(i -> (i == 1 ? t : 2t), d)) for t in ts]
end

# sinusoid: [sin(2π·1·t), sin(2π·2·t), ...]
function make_path_sin(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    ω = 2π
    [SVector{d,Float64}(ntuple(i -> sin(ω * i * t), d)) for t in ts]
end

function make_path(d::Int, N::Int, kind::Symbol)
    if kind === :linear
        return make_path_linear(d, N)
    elseif kind === :sin
        return make_path_sin(d, N)
    else
        error("Unknown path_kind: $kind (expected :linear or :sin)")
    end
end

# -------- one benchmark case (Julia only) --------

function bench_case(d::Int, m::Int, N::Int, path_kind::Symbol, op::Symbol, repeats::Int)
    path = make_path(d, N, path_kind)
    tensor_type = Chen.Tensor{eltype(path[1])}

    # Helper closures for the two operations
    run_sig() = signature_path(tensor_type, path, m)
    run_logsig() = Chen.log(signature_path(tensor_type, path, m))

    # Select function and warmup
    if op === :signature
        run_sig()
        # Note: We must interpolate local functions with $ for BenchmarkTools
        t_jl = @belapsed $run_sig() evals=1 samples=repeats
        a_jl = @allocated run_sig()
    elseif op === :logsignature
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
            t_ms = t_ms,
            alloc_KiB = alloc_KiB)
end

# -------- sweep + write grid to file --------

function run_bench()
    cfg = load_config()
    Ns, Ds, Ms = cfg.Ns, cfg.Ds, cfg.Ms
    path_kind  = cfg.path_kind
    runs_dir   = cfg.runs_dir
    repeats    = cfg.repeats
    operations = cfg.operations

    println("Running Julia benchmark with config:")
    println("  path_kind  = $path_kind")
    println("  Ns         = $(Ns)")
    println("  Ds         = $(Ds)")
    println("  Ms         = $(Ms)")
    println("  operations = $(operations)")
    println("  runs_dir   = \"$runs_dir\"")
    println("  repeats    = $repeats")

    results = NamedTuple[]

    for N in Ns, d in Ds, m in Ms, op in operations
        push!(results, bench_case(d, m, N, path_kind, op, repeats))
    end

    runs_path = joinpath(@__DIR__, runs_dir)
    isdir(runs_path) || mkpath(runs_path)

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    file = joinpath(runs_path, "run_julia_$ts.csv")

    header = ["N", "d", "m", "path_kind", "operation", "t_ms", "alloc_KiB"]
    data = Array{Any}(undef, length(results) + 1, length(header))
    data[1, :] = header

    for (i, r) in enumerate(results)
        data[i + 1, 1] = r.N
        data[i + 1, 2] = r.d
        data[i + 1, 3] = r.m
        data[i + 1, 4] = String(r.path_kind)
        data[i + 1, 5] = String(r.operation)
        data[i + 1, 6] = r.t_ms
        data[i + 1, 7] = r.alloc_KiB
    end

    writedlm(file, data, ',')

    println("============================================================")
    println("Benchmark grid written to: $file")
    return file
end

run_bench()