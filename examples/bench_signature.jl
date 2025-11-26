using Chen
using StaticArrays
using BenchmarkTools

# --- Old version (your original) for comparison ---
function signature_path_old!(
    out::AT,
    path::Vector{SVector{D,T}},
) where {D,T, AT <: Chen.AbstractTensor{T}}

    d = D
    a = similar(out)
    b = similar(out)
    segment_tensor = similar(out)
    displacement = Vector{T}(undef, d)

    displacement .= path[2] - path[1]
    Chen.exp!(a, displacement)

    for i in 2:length(path)-1
        displacement .= path[i+1] - path[i]
        Chen.exp!(segment_tensor, displacement)
        Chen.mul!(b, a, segment_tensor)
        a, b = b, a
    end

    return a
end

# --- New in-place version (writes result into `out`) ---
function signature_path_new!(
    out::AT,
    path::Vector{SVector{D,T}},
) where {D,T, AT <: Chen.AbstractTensor{T}}

    @assert length(path) ≥ 2 "path must have at least 2 points"

    a = out
    b = similar(out)
    segment_tensor = similar(out)

    @inbounds begin
        Δ = path[2] - path[1]
        Chen.exp!(a, Δ)

        for i in 2:length(path)-1
            Δ = path[i+1] - path[i]
            Chen.exp!(segment_tensor, Δ)
            Chen.mul!(b, a, segment_tensor)
            a, b = b, a
        end
    end

    # Ensure final result is stored in `out`
    if a !== out
        copy!(out, a)
    end

    return out
end

# --- Build a path with the right element type: Vector{SVector{D,Float64}} ---

function build_path(::Val{D}, n::Int) where {D}
    ts = range(0.0, 1.0; length=n)
    path = Vector{SVector{D,Float64}}(undef, n)
    @inbounds for (i, t) in enumerate(ts)
        # here just a simple diagonal path: (t, t, ..., t)
        path[i] = SVector{D,Float64}(ntuple(_ -> t, D))
    end
    return path
end

# --- Benchmark harness ---
function bench_signature()
    D = 3          # dimension
    m = 4          # truncation level
    n = 1_000      # number of time steps

    path = build_path(Val(D), n)

    out_old = Chen.Tensor{Float64}(D, m)
    out_new = Chen.Tensor{Float64}(D, m)

    println("Benchmarking allocating API (baseline):")
    @btime Chen.signature_path(Chen.Tensor{Float64}, $path, $m);

    println("\nBenchmarking old in-place version (returns new Tensor, `out` unused):")
    @btime signature_path_old!($out_old, $path);

    println("\nBenchmarking new in-place version (writes into `out`):")
    @btime signature_path_new!($out_new, $path);
end

bench_signature()
