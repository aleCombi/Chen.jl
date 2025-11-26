######################## examples/bench_gpt.jl ########################
#
# One-shot comparison of Chen.signature_path!:
#
#   1) Baseline (original Chen.exp!)
#   2) After patching Chen.exp!(::Tensor, ::AbstractVector)
#
# Usage from REPL:
#
#     include("examples/bench_gpt.jl")
#
#######################################################################

using Chen
using StaticArrays
using LoopVectorization: @avx
using BenchmarkTools

const Tensor = Chen.Tensor

# --------------------------------------------------------------------
# 1. Benchmark helper
# --------------------------------------------------------------------

function _bench_signature(label; d::Int = 3, m::Int = 4, N::Int = 10_000)
    println(label)
    println("  d = $d, m = $m, N = $N")

    path = [@SVector randn(d) for _ in 1:N]
    out  = Tensor{Float64}(d, m)

    # warmup
    Chen.signature_path!(out, path)

    t = @belapsed Chen.signature_path!($out, $path)

    println("  time: $(round(t*1e3; digits=3)) ms")
    println()
    return t
end

# --------------------------------------------------------------------
# 2. Run baseline BEFORE patching
# --------------------------------------------------------------------

println("=====================================================")
println(" Baseline Chen.signature_path! (original Chen.exp!)  ")
println("=====================================================\n")

baseline_time = _bench_signature("Baseline:"; d=3, m=4, N=10_000)

# --------------------------------------------------------------------
# 3. Define fast kernel and patch Chen.exp!
# --------------------------------------------------------------------

@inline function _segment_level_offsets_fast!(
    out::Vector{T}, Δ::AbstractVector{T}, scale::T,
    prev_start::Int, prev_len::Int, cur_start::Int
) where {T}
    d = length(Δ)
    @inbounds for i in 1:d
        s    = scale * Δ[i]                    # hoisted once per i
        base = cur_start + (i - 1) * prev_len - 1
        @avx for j in 1:prev_len               # contiguous, SIMD-friendly
            out[base + j] = s * out[prev_start + j - 1]
        end
    end
    return nothing
end

"""
    Chen.exp!(out::Chen.Tensor{T}, x::AbstractVector{T})

Patched fast-path for tensor exponential on Chen.Tensor backend.
This replaces the existing method once this file is included.
"""
function Chen.exp!(out::Tensor{T}, x::AbstractVector{T}) where {T}
    d = out.dim
    m = out.level
    offsets = out.offsets
    coeffs  = out.coeffs

    @assert length(x) == d

    @inbounds begin
        # level-0 unit
        coeffs[offsets[1] + 1] = one(T)
        m == 0 && return nothing

        # level 1
        idx        = offsets[2] + 1
        curlen     = d
        prev_start = idx
        copyto!(coeffs, idx, x, 1, d)
        idx += curlen

        m == 1 && return nothing

        # levels 2..m
        for level in 2:m
            prev_len  = curlen
            curlen   *= d
            cur_start = idx
            scale     = inv(T(level))

            _segment_level_offsets_fast!(
                coeffs, x, scale,
                prev_start, prev_len, cur_start
            )

            prev_start = cur_start
            idx += curlen
        end
    end

    return nothing
end

# --------------------------------------------------------------------
# 4. Benchmark AFTER patching
# --------------------------------------------------------------------

println("=====================================================")
println(" After patch: Chen.signature_path! (patched exp!)    ")
println("=====================================================\n")

patched_time = _bench_signature("Patched:"; d=3, m=4, N=10_000)

# --------------------------------------------------------------------
# 5. Report speedup
# --------------------------------------------------------------------

speedup = baseline_time / patched_time
println("=====================================================")
println(" Summary                                            ")
println("=====================================================")
println("  Baseline: $(round(baseline_time*1e3; digits=3)) ms")
println("  Patched : $(round(patched_time*1e3; digits=3)) ms")
println("  Speedup : ×$(round(speedup; digits=3))")
println("=====================================================")

###################### end of bench_gpt.jl ###########################
