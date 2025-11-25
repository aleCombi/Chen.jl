# sigcheck.jl
#
# Usage:
#   julia --project=. sigcheck.jl N d m path_kind operation
#
# Arguments:
#   operation: "signature" or "logsignature"
#
# Output:
#   Prints the coefficients (in the iisignature-compatible layout)
#   as a single space-separated line on stdout.

using StaticArrays
using PathSignatures
using LinearAlgebra

# We need the Lyndon basis utilities to project the tensor log
# onto the Lyndon basis for comparison with iisignature.
# Since these are not exported by PathSignatures, we include the file.
include(joinpath(@__DIR__, "..", "src", "lyndon_basis.jl"))

if length(ARGS) < 5
    error("Usage: julia sigcheck.jl N d m path_kind operation")
end

N = parse(Int, ARGS[1])
d = parse(Int, ARGS[2])
m = parse(Int, ARGS[3])
path_kind = Symbol(ARGS[4]) # "linear" or "sin"
operation = Symbol(ARGS[5]) # "signature" or "logsignature"

# -------- path generators --------

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
tensor_type = PathSignatures.Tensor{eltype(path[1])}

# 1. Compute Signature
sig = signature_path(tensor_type, path, m)

output_vec = Float64[]

if operation === :signature
    # Compare raw signature coefficients (levels 1..m)
    # iisignature flattens them into a single vector.
    # Julia's Tensor stores them with padding, so we extract valid regions.
    
    for k in 1:m
        start_idx = sig.offsets[k+1] + 1
        len = d^k
        append!(output_vec, view(sig.coeffs, start_idx : start_idx+len-1))
    end

elseif operation === :logsignature
    # 2. Compute Log Signature
    log_sig_tensor = PathSignatures.log(sig)
    
    # 3. Project to Lyndon basis
    # (iisignature.logsig returns coeffs in the Lyndon basis)
    lynds, L, _ = build_L(d, m)
    
    # project_to_lyndon returns the vector of coefficients
    output_vec = project_to_lyndon(log_sig_tensor, lynds, L)

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