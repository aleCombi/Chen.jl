using Revise, PathSignatures, PythonCall
@py import iisignature
@py import numpy as np
using BenchmarkTools

# Define path
using StaticArrays
f(t) = @SVector [t, 2t]

ts = range(0, stop=1, length=1000)  # 9 segments, 10 points
m = 7
d = length(f(0.0))

# Julia: N-point signature via Chen
sig_julia = signature_path(f, ts, m)

# Python: full path signature
path = reduce(vcat, [f(t)' for t in ts])  # Matrix (10, d) flattened row-wise
path_np = np.asarray(path; order="C")
sig_py = iisignature.sig(path_np, m)
sig_py_julia = pyconvert(Vector{Float64}, sig_py)

# ✅ Validate match
@assert isapprox(sig_julia, sig_py_julia; atol=1e-12)

# 🕒 Benchmark
println("Benchmarking:")
@btime signature_path($f, $ts, $m)
@btime $iisignature.sig($path_np, $m)