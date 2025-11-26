using StaticArrays
using LoopVectorization: @avx, @turbo

struct Tensor{T} <: AbstractTensor{T}
    coeffs::Vector{T}
    dim::Int
    level::Int
    offsets::Vector{Int}
    
    function Tensor(coeffs::Vector{T}, dim::Int, level::Int) where {T}
        offsets = level_starts0(dim, level)
        new{T}(coeffs, dim, level, offsets)
    end

    function Tensor{T}(dim::Int, level::Int) where {T}
        offsets = level_starts0(dim, level)     # includes pad + end sentinel
        coeffs  = Vector{T}(undef, offsets[end])
        new{T}(coeffs, dim, level, offsets)
    end
end

# -------- Dense ↔ Dense (respects per-level padding) --------
function Base.isapprox(a::Chen.Tensor{Ta}, b::Chen.Tensor{Tb};
                       atol::Real=1e-8, rtol::Real=1e-8) where {Ta,Tb}
    a.dim == b.dim && a.level == b.level || return false
    sA, sB = a.offsets, b.offsets
    d, m = a.dim, a.level
    len = 1                     # = d^k
    @inbounds begin
        # level-0
        a0 = a.coeffs[sA[1] + 1]; b0 = b.coeffs[sB[1] + 1]
        if !(abs(a0 - b0) <= atol + rtol*max(abs(a0),abs(b0))); return false; end
        # levels 1..m
        for k in 1:m
            astart = sA[k + 1] + 1
            bstart = sB[k + 1] + 1
            for j in 0:len-1
                va = a.coeffs[astart + j]; vb = b.coeffs[bstart + j]
                if !(abs(va - vb) <= atol + rtol*max(abs(va),abs(vb))); return false; end
            end
            len *= d
        end
    end
    return true
end

## Tensor interface

Base.length(ts::Tensor{T}) where T = length(ts.coeffs)
Base.getindex(ts::Tensor{T}, i::Int) where T = ts.coeffs[i]
Base.show(io::IO, ts::Tensor{T}) where T =
    print(io, "Tensor(dim=$(ts.dim), level=$(ts.level), length=$(length(ts)))")

    # Element type of a Tensor
Base.eltype(::Tensor{T}) where {T} = T

# Allocate a new Tensorwith the same "shape" (dim, level)
Base.similar(ts::Tensor{T}) where {T} = Tensor{T}(ts.dim, ts.level)

# Same, but change element type (handy for promoting to BigFloat, Dual, etc.)
Base.similar(ts::Tensor{T}, ::Type{S}) where {T,S} = Tensor{S}(ts.dim, ts.level)

# Allocate-and-copy convenience
Base.copy(ts::Tensor{T}) where {T} = Tensor(copy(ts.coeffs), ts.dim, ts.level)

# In-place copy with shape check (future-proof for pipelines)
function Base.copy!(dest::Tensor, src::Tensor)
    @assert dest.dim == src.dim && dest.level == src.level "Tensor shape mismatch"
    copyto!(dest.coeffs, src.coeffs)
    return dest
end

dim(ts::Tensor)   = ts.dim
level(ts::Tensor) = ts.level
coeffs(ts::Tensor) = ts.coeffs

# --- small helpers ------------------------------------------------------------

@inline function _zero!(ts::Tensor{T}) where {T}
    fill!(ts.coeffs, zero(T)); ts
end

@inline function add_scaled!(dest::Tensor{T}, src::Tensor{T}, α::T) where {T}
    @inbounds @simd for i in eachindex(dest.coeffs, src.coeffs)
        dest.coeffs[i] = muladd(α, src.coeffs[i], dest.coeffs[i])
    end
    dest
end

@inline _write_unit!(t::Tensor{T}) where {T} =
    (t.coeffs[t.offsets[1] + 1] = one(T); t)

function lift1!(ts::Tensor{T}, x::AbstractVector{T}) where {T}
    @assert length(x) == ts.dim
    fill!(ts.coeffs, zero(T))
    @inbounds copyto!(ts.coeffs, 1, x, 1, ts.dim)  # write level-1 block
    return ts
end

"""
    exp!(out, x::AbstractVector)

Compute the truncated tensor exponential of a level-1 element `x`:
    exp(x) = x + x^2/2! + ... + x^m/m!
(level-0 unit is implicit/not stored). Works for any Tensor backend.
"""
function exp!(out::Tensor{T}, x::AbstractVector{T}) where {T}
    # Optimized SIMD path for Vector input into Dense Tensor
    d = out.dim
    m = out.level
    out.coeffs[out.offsets[1] + 1] = one(T)
    m == 0 && return nothing

    idx    = out.offsets[2] + 1
    curlen = d
    copyto!(out.coeffs, idx, x, 1, d)
    prev_start = idx
    idx += curlen

    if m == 1
        return nothing
    end

    @inbounds for level in 2:m
        prev_len  = curlen
        curlen   *= d
        cur_start = idx
        scale = inv(T(level))
        _segment_level_offsets!(out.coeffs, x, scale,
                                prev_start, prev_len, cur_start)
        prev_start = cur_start
        idx += curlen
    end
    return nothing
end

@inline function mul_grouplike!(
    out_tensor::Tensor{T}, x1_tensor::Tensor{T}, x2_tensor::Tensor{T}
) where {T}
    out = out_tensor.coeffs
    x1  = x1_tensor.coeffs
    x2  = x2_tensor.coeffs

    m = out_tensor.level
    d = out_tensor.dim
    offsets = out_tensor.offsets

    # level-0 index
    i0 = offsets[1] + 1

    # @assert x1[i0] == one(T) "mul_grouplike!: expected x1 level-0 == 1"
    # @assert x2[i0] == one(T) "mul_grouplike!: expected x2 level-0 == 1"

    # level-0: 1 * 1 = 1
    out[i0] = one(T)

    out_len = d

    @inbounds for k in 1:m
        out_start = offsets[k + 1] + 1

        # Base: out_k = x1_k + x2_k
        @avx for j in 0:out_len-1
            out[out_start + j] = x1[out_start + j] + x2[out_start + j]
        end

        # Middle terms: i = 1..k-1
        a_len = d
        for i in 1:(k-1)
            a_block_start = offsets[i + 1]
            b_block_start = offsets[k - i + 1]
            b_len = out_len ÷ a_len

            @avx for ai in 1:a_len, bi in 1:b_len
                row0 = out_start + (ai - 1) * b_len - 1
                out[row0 + bi] = muladd(
                    x1[a_block_start + ai],
                    x2[b_block_start + bi],
                    out[row0 + bi],
                )
            end

            a_len *= d
        end

        out_len *= d
    end

    return out_tensor
end

# generic path: arbitrary first coefficients (a0, b0)
@inline function mul!(
    out_tensor::Tensor{T}, x1_tensor::Tensor{T}, x2_tensor::Tensor{T}
) where {T}
    out, x1, x2, m = out_tensor.coeffs, x1_tensor.coeffs, x2_tensor.coeffs, out_tensor.level
    offsets = out_tensor.offsets
    d = out_tensor.dim

    a0 = x1[offsets[1] + 1]
    b0 = x2[offsets[1] + 1]
    out[offsets[1] + 1] = a0 * b0

    out_len = d
    @inbounds for k in 1:m
        out_start = offsets[k + 1] + 1

        # i = 0: out_k ← a0 * x2_k  (write-only; no zeroing)
        b_start = offsets[k + 1]
        if a0 == one(T)
            copyto!(out, out_start, x2, b_start + 1, out_len)
        elseif a0 == zero(T)
            @avx for j in 1:out_len
                out[out_start + j - 1] = zero(T)
            end
        else
            @avx for j in 1:out_len
                out[out_start + j - 1] = a0 * x2[b_start + j]
            end
        end

        # middle terms: i = 1..k-1
        a_len = d
        for i in 1:(k-1)
            a_start = offsets[i + 1]
            b_start = offsets[k - i + 1]
            b_len   = out_len ÷ a_len

            @avx for ai in 1:a_len, bi in 1:b_len
                row0 = out_start + (ai - 1) * b_len - 1
                out[row0 + bi] = muladd(x1[a_start + ai], x2[b_start + bi], out[row0 + bi])
            end

            a_len *= d
        end

        # i = k: out_k += b0 * x1_k
        if b0 != zero(T)
            a_start = offsets[k + 1]
            if b0 == one(T)
                @avx for j in 1:out_len
                    out[out_start + j - 1] += x1[a_start + j]
                end
            else
                @avx for j in 1:out_len
                    out[out_start + j - 1] = muladd(b0, x1[a_start + j], out[out_start + j - 1])
                end
            end
        end

        out_len *= d
    end
    return out
end


@inline function _segment_level_offsets!(
    out::Vector{T}, Δ::AbstractVector{D}, scale::T,
    prev_start::Int, prev_len::Int, cur_start::Int
) where {T,D}
    d = length(Δ)
    @inbounds @avx for i in 1:d, j in 1:prev_len
        s = scale * Δ[i]
        base = cur_start + (i - 1) * prev_len - 1
        out[base + j] = s * out[prev_start + j - 1]
    end
    return nothing
end

"""
    level_starts0(dim::Int, level::Int) -> Vector{Int}

Return a vector `s` of length `level + 1` with **0-based** start indices for the
flattened tensor-series blocks of each level `k = 1..level` in column-major layout.

Invariant:
- `s[1] == 0`
- `s[k+1] - s[k] == dim^k` (length of level-`k` block)
- `s[level+1] == (dim^(level+1) - dim) ÷ (dim - 1)` (total length)
"""
function level_starts0(d, m)
    offsets = Vector{Int}(undef, m + 2)
    offsets[1] = 0
    len = 1
    @inbounds for k in 1:m+1
        offsets[k+1] = offsets[k] + len
        len *= d
    end
    # pad so level-1 (offsets[2]+1) is 64B-aligned for Float64
    W = 8
    pad = (W - (offsets[2] % W)) % W
    if pad != 0
        @inbounds for k in 2:length(offsets)
            offsets[k] += pad
        end
    end
    return offsets
end

"""
    log!(out::Tensor{T}, g::Tensor{T}) where T

Compute the truncated tensor logarithm of a group-like element `g` up to `out.level`,
using the power-series:
    log(g) = (g-1) - (g-1)^2/2 + (g-1)^3/3 - ...  (truncated at level m)
All multiplications are done with `mul!`. The result has zero level-0.
"""
function log!(out::Tensor{T}, g::Tensor{T}) where {T}
    @assert out.dim == g.dim && out.level == g.level "log!: shape mismatch"
    m = out.level
    m == 0 && (fill!(out.coeffs, zero(T)); return out)

    offsets = out.offsets
    i0 = offsets[1] + 1

    # Require group-like normalisation at level-0
    @assert g.coeffs[i0] == one(T) "log!: expected level-0 == 1 (group-like element)"

    # X := g - 1
    X = similar(out)
    copy!(X, g)
    X.coeffs[i0] -= one(T)        # make level-0 of X zero

    # Prepare accumulators/buffers
    _zero!(out)                   # out ← 0 (and we'll ensure out.level-0 stays 0)
    P = similar(out)              # current power (starts as X)
    Q = similar(out)              # scratch for next power
    copy!(P, X)

    sgn = one(T)                  # (+1, -1, +1, ...)
    for k in 1:m
        # out += ((-1)^(k+1) / k) * P
        add_scaled!(out, P, sgn / T(k))

        # Next power if needed: P ← P * X
        if k < m
            mul!(Q, P, X)
            Q.coeffs[i0] = zero(T)  # enforce zero level-0 (should already be)
            P, Q = Q, P             # swap buffers
        end
        sgn = -sgn
    end

    out.coeffs[i0] = zero(T)      # ensure level-0 is 0 for a Lie element
    return out
end

"""
    log(g::Tensor{T}) -> Tensor{T}

Allocating wrapper for `log!`.
"""
function log(g::Tensor{T}) where {T}
    out = similar(g)
    return log!(out, g)
end