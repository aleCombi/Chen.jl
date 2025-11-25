using Revise, PythonCall, BenchmarkTools, StaticArrays, Chen, CondaPkg
CondaPkg.add_pip("numpy")
CondaPkg.add_pip("iisignature")
@py import iisignature
@py import numpy as np

# path: R -> R^2
f(t) = @SVector [t, 2t]

ts = range(0.0, stop=1.0, length=10000)
m = 5

# --- Julia signatures ---

path = [f(t) for t in ts]               # precomputed path
sig_julia_vec = signature_path(Tensor{typeof(path[1][1])}, path, m)

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
@assert isapprox(sig_julia_vec.coeffs[sig_julia_vec.offsets[2]+1:end],  sig_py_julia; atol=1e-6)

dense_type = Tensor{typeof(path[1][1])}
sparse_type = SparseTensor{typeof(path[1][1])}

# 🕒 Benchmark
println("Benchmarking:")
println("Julia Dense (signature_path(path, m))")
@btime signature_path($dense_type, $path, $m);
println("Julia Sparse (signature_path(path, m))")
@btime signature_path($sparse_type, $path, $m);

