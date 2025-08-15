using Revise, PythonCall, BenchmarkTools, StaticArrays
@py import iisignature
@py import numpy as np

# path: R -> R^2
f(t) = @SVector [t, 2t, 3.5t]

ts = range(0.0, stop=1.0, length=1000)
m = 4

# --- Julia signatures ---

path = [f(t) for t in ts]               # precomputed path
sig_julia_vec = signature_path(path, m)

# --- Python iisignature baseline ---
# build a compact C-contiguous matrix without tons of temporaries
d = length(first(path))
path_mat = Array{Float64}(undef, length(ts), d)
@inbounds for i in eachindex(ts)
    v = path[i]
    for j in eachindex(v)
        path_mat[i,j] = v[j]
    end
end
path_np = np.asarray(path_mat; order="C")
sig_py = iisignature.sig(path_np, m)
sig_py_julia = pyconvert(Vector{Float64}, sig_py)

# ✅ Validate match
@assert isapprox(sig_julia_vec,  sig_py_julia; atol=1e-12)

# 🕒 Benchmark
println("Benchmarking:")
println("Julia (signature_path(path, m))")
@btime signature_path($path, $m);
println("Python (iisignature.sig)")
@btime $iisignature.sig($path_np, $m);
