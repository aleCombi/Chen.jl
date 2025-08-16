# examples/benchmark.jl

using Revise, PythonCall, BenchmarkTools, StaticArrays, PathSignatures
@py import iisignature
@py import numpy as np

# -------- path generators (no extra deps) --------
# linear: [t, 2t, 2t, ...]
make_path_linear(d::Int, N::Int) = begin
    ts = range(0.0, stop=1.0, length=N)
    [SVector{d,Float64}(ntuple(i -> (i==1 ? t : 2t), d)) for t in ts]
end

# sinusoid: [sin(2π·1·t), sin(2π·2·t), ...]
make_path_sin(d::Int, N::Int) = begin
    ts = range(0.0, stop=1.0, length=N)
    ω = 2π
    [SVector{d,Float64}(ntuple(i -> sin(ω*i*t), d)) for t in ts]
end

# Build a compact C-contiguous matrix for Python
to_numpy_matrix(path::Vector{<:StaticVector}) = begin
    N = length(path); d = length(path[1])
    mat = Array{Float64}(undef, N, d)
    @inbounds for i in 1:N
        v = path[i]
        @inbounds for j in 1:d
            mat[i,j] = v[j]
        end
    end
    np.asarray(mat; order="C")
end

# -------- one benchmark case --------
function bench_case(d::Int, m::Int, N::Int, kind::Symbol)
    path = kind === :linear ? make_path_linear(d, N) : make_path_sin(d, N)

    # warmups
    sig_julia = signature_path(path, m)
    path_np = to_numpy_matrix(path)
    sig_py   = iisignature.sig(path_np, m)

    # validate
    sig_py_vec = pyconvert(Vector{Float64}, sig_py)
    @assert isapprox(sig_julia, sig_py_vec; atol=1e-8, rtol=1e-8)

    # timings (no `$` interpolation needed inside a function)
    t_jl = @belapsed signature_path(path, m)
    a_jl = @allocated signature_path(path, m)
    t_py = @belapsed iisignature.sig(path_np, m)

    println("—"^60)
    println("d=$d, m=$m, N=$N, kind=$kind")
    @printf "Julia:   %8.3f ms   allocations: %7.1f KiB\n" (t_jl*1000) (a_jl/1024)
    @printf "Python:  %8.3f ms   (iisignature)\n" (t_py*1000)
    @printf "Speedup (Julia / Python): ×%.2f\n" (t_py / t_jl)
    return nothing
end

# -------- sweep --------
function run_bench()
    Ns   = [10_000]          # adjust if you like
    Ds   = [2, 3, 8]         # dimensions
    Ms   = [4, 5, 6]         # truncation levels
    Kinds = [:linear, :sin]  # path families

    for N in Ns, d in Ds, m in Ms, kind in Kinds
        bench_case(d, m, N, kind)
    end
    println("Done.")
end

run_bench()
