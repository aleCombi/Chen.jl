# benchmark/sigcheck.jl

using StaticArrays
using ChenSignatures
using LinearAlgebra

if length(ARGS) < 5
    error("Usage: julia sigcheck.jl N d m path_kind operation")
end

const N = parse(Int, ARGS[1])
const d = parse(Int, ARGS[2])
const m = parse(Int, ARGS[3])
const path_kind = Symbol(ARGS[4]) # "linear" or "sin"
const operation = Symbol(ARGS[5]) # "signature" or "logsignature"

# --- Path Generators (Same logic as benchmark.jl) ---

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

# --- Main Logic ---

path = make_path(d, N, path_kind)
# Determine element type from path
tensor_type = ChenSignatures.Tensor{eltype(path[1])}

# 1. Compute Signature
sig = signature_path(tensor_type, path, m)

output_vec = Float64[]

if operation === :signature
    # Compare raw signature coefficients (levels 1..m)
    # The Tensor type stores level 0 (scalar 1.0) at the start, we skip it.
    for k in 1:m
        start_idx = sig.offsets[k+1] + 1
        len = d^k
        append!(output_vec, view(sig.coeffs, start_idx : start_idx+len-1))
    end

elseif operation === :logsignature
    # 2. Compute Logarithm
    log_sig_tensor = ChenSignatures.log(sig)
    
    # 3. Project to Lyndon basis
    # 'build_L' and 'project_to_lyndon' are now exported directly by ChenSignatures
    lynds, L, _ = build_L(d, m)
    output_vec = project_to_lyndon(log_sig_tensor, lynds, L)

else
    error("Unknown operation: $operation")
end

# --- Output for Python ---

# DEBUG info to stderr (so it doesn't pollute stdout for parsing)
println(stderr, "=== sigcheck debug ===")
println(stderr, "N=$N, d=$d, m=$m, kind=$path_kind, op=$operation")
println(stderr, "output length = ", length(output_vec))
println(stderr, "======================")

# Numeric output to stdout
for (i, v) in enumerate(output_vec)
    print(v)
    if i < length(output_vec)
        print(' ')
    end
end
println()