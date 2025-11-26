#############################################
# examples/linear_update_benchmark.jl
#############################################

using StaticArrays
using BenchmarkTools
using Random
using LoopVectorization: @avx
using ChenSignatures

# ------------------------------------------------------------------
# Parameters / backend type
# ------------------------------------------------------------------

const D = 5             # path dimension
const M = 5             # signature level
const T = Float64

# Concrete ChenSignatures backend with fixed D,M
const AT = ChenSignatures.Tensor{T, D, M}

println("Benchmarking Type: $AT")

# ------------------------------------------------------------------
# Fused linear-segment update:
#   sig ← sig ⊗ exp(Δ)
# ------------------------------------------------------------------

"""
    update_signature_linear!(sig, Δ, seg, scratch)

Update the group-like signature `sig` in-place with a single linear segment increment `Δ`,
i.e. perform

    sig ← sig ⊗ exp(Δ)

Arguments:
- `sig::AT`: current signature (group-like, level-0 = 1)
- `Δ::AbstractVector{T}`: segment increment of length `D`
- `seg::AT`: scratch tensor to hold `exp(Δ)`
- `scratch::AT`: scratch tensor to hold a copy of the old signature
"""
function update_signature_linear!(
    sig::AT,
    Δ::AbstractVector{T},
    seg::AT,
    scratch::AT,
)
    @assert length(Δ) == D

    # 1) S := sig
    copy!(scratch, sig)

    # 2) E := exp(Δ)
    ChenSignatures.exp!(seg, Δ)

    S = scratch.coeffs
    E = seg.coeffs
    out = sig.coeffs
    offs = sig.offsets
    d = D
    m = M

    # 3) Level-0 stays 1 (group-like)
    i0 = offs[1] + 1
    @assert S[i0] == one(T)
    @assert E[i0] == one(T)
    out[i0] = one(T)

    # 4) Levels 1..M: out_k = S_k + E_k + Σ_{i=1}^{k-1} S_i ⊗ E_{k-i}
    out_len = d

    @inbounds for k in 1:m
        out_start = offs[k + 1] + 1

        # Level-k starts of S and E blocks
        S_k_start = offs[k + 1] + 1
        E_k_start = offs[k + 1] + 1

        # Base: out_k = S_k + E_k
        @avx for j in 0:out_len-1
            out[out_start + j] = S[S_k_start + j] + E[E_k_start + j]
        end

        # Cross terms: i = 1..k-1:  S_i ⊗ E_{k-i}
        a_len = d
        for i in 1:(k-1)
            S_i_start = offs[i + 1]
            E_j_start = offs[k - i + 1]
            b_len = out_len ÷ a_len

            @avx for ai in 1:a_len, bi in 1:b_len
                row0 = out_start + (ai - 1) * b_len - 1
                out[row0 + bi] = muladd(
                    S[S_i_start + ai],
                    E[E_j_start + bi],
                    out[row0 + bi],
                )
            end

            a_len *= d
        end

        out_len *= d
    end

    return sig
end

# ------------------------------------------------------------------
# Fused signature_path! using update_signature_linear!
# ------------------------------------------------------------------

"""
    signature_path_linear!(out, path)

Compute the signature of a piecewise-linear path `path` into `out`,
using the fused linear update `update_signature_linear!`.

Backend is `ChenSignatures.Tensor{T,D,M}` with fixed `D,M`.
"""
function signature_path_linear!(
    out::AT,
    path::Vector{SVector{D,T}},
)
    @assert length(path) ≥ 2 "path must have at least 2 points"

    # Scratch buffers
    seg     = similar(out)  # for exp(Δ)
    scratch = similar(out)  # for old signature

    sig = out

    @inbounds begin
        # First segment: signature of the first segment alone is exp(Δ₁)
        Δ = path[2] - path[1]
        ChenSignatures.exp!(sig, Δ)

        # Remaining segments: use fused linear update
        for i in 2:length(path)-1
            Δ = path[i+1] - path[i]
            update_signature_linear!(sig, Δ, seg, scratch)
        end
    end

    return out
end

# ------------------------------------------------------------------
# Benchmarks: ChenSignatures.signature_path! vs fused signature_path_linear!
# ------------------------------------------------------------------

function run_benchmarks(; Nseg::Int = 1000)
    N = Nseg + 1
    rng = MersenneTwister(1234)

    # Random path
    path = [@SVector rand(rng, T, D) for _ in 1:N]

    # Output buffers (use generic ctor to get the right concrete type)
    out_chen  = ChenSignatures.Tensor{T}(D, M)  # should be Tensor{T,D,M}
    out_fused = ChenSignatures.Tensor{T}(D, M)

    println("Nseg = $Nseg")

    # Warmup + correctness check
    ChenSignatures.signature_path!(out_chen, path)
    signature_path_linear!(out_fused, path)

    @assert isapprox(out_chen, out_fused; atol=1e-8, rtol=1e-8) \
        "Fused implementation disagrees with ChenSignatures.signature_path!"

    println("\n=== ChenSignatures.signature_path! ===")
    @btime ChenSignatures.signature_path!($out_chen, $path)

    println("\n=== signature_path_linear! (fused update) ===")
    @btime signature_path_linear!($out_fused, $path)

    # Also report per-segment times using @belapsed
    t_chen  = @belapsed ChenSignatures.signature_path!($out_chen, $path)
    t_fused = @belapsed signature_path_linear!($out_fused, $path)

    println("\n--- Summary (Nseg = $Nseg) ---")
    println("ChenSignatures.signature_path!:     $(t_chen * 1e3) ms total  ($(t_chen / Nseg * 1e6) μs/segment)")
    println("signature_path_linear!:   $(t_fused * 1e3) ms total  ($(t_fused / Nseg * 1e6) μs/segment)")

    return nothing
end

run_benchmarks()
