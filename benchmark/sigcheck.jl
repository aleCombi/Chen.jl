# benchmark/sigcheck.jl

using StaticArrays
using Chen
using LinearAlgebra

if length(ARGS) < 5
    error("Usage: julia sigcheck.jl N d m path_kind operation")
end

N = parse(Int, ARGS[1])
d = parse(Int, ARGS[2])
m = parse(Int, ARGS[3])
path_kind = Symbol(ARGS[4]) # "linear" or "sin"
operation = Symbol(ARGS[5]) # "signature" or "logsignature"

function make_path_linear(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    [SVector{d,Float64}(ntuple(i -> (i == 1 ? t : 2t), d)) for t in ts]
end

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

# -------- main logic --------

path = make_path(d, N, path_kind)
tensor_type = Chen.Tensor{eltype(path[1])}

# 1. Compute Signature
sig = signature_path(tensor_type, path, m)

output_vec = Float64[]

if operation === :signature
    # Compare raw signature coefficients (levels 1..m)
    for k in 1:m
        start_idx = sig.offsets[k+1] + 1
        len = d^k
        append!(output_vec, view(sig.coeffs, start_idx : start_idx+len-1))
    end

elseif operation === :logsignature
    # 2. Compute Log Signature
    log_sig_tensor = Chen.log(sig)
    
    # 3. Project to Lyndon basis
    # FIXED: Accessing Algebra submodule
    lynds, L, _ = Chen.Algebra.build_L(d, m)
    
    # FIXED: Accessing Algebra submodule
    output_vec = Chen.Algebra.project_to_lyndon(log_sig_tensor, lynds, L)

else
    error("Unknown operation: $operation")
end

# DEBUG info to stderr
println(stderr, "=== sigcheck debug ===")
println(stderr, "N=$N, d=$d, m=$m, kind=$path_kind, op=$operation")
println(stderr, "output length = ", length(output_vec))
println(stderr, "======================")

# ---------- NUMERIC OUTPUT TO STDOUT ----------
for (i, v) in enumerate(output_vec)
    print(v)
    if i < length(output_vec)
        print(' ')
    end
end
println()