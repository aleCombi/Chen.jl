using StaticArrays
using LoopVectorization: @avx, @turbo

# -------------------------------------------------------------------
# Tensor type: specialize on level M, keep dim runtime
# -------------------------------------------------------------------

struct Tensor{T,M} <: AbstractTensor{T}
    coeffs::Vector{T}
    dim::Int
    offsets::Vector{Int}
end

# Backwards-compatible access: ts.level
Base.getproperty(ts::Tensor{T,M}, s::Symbol) where {T,M} =
    s === :level ? M : getfield(ts, s)

# Public, generic constructor (used by existing code)
function Tensor(coeffs::Vector{T}, dim::Int, level::Int) where {T}
    offsets = level_starts0(dim, level)
    return Tensor{T,level}(coeffs, dim, offsets)
end

# Allocating constructor: Tensor{T}(dim, level)
function Tensor{T}(dim::Int, level::Int) where {T}
    offsets = level_starts0(dim, level)
    coeffs  = Vector{T}(undef, offsets[end])
    return Tensor{T,level}(coeffs, dim, offsets)
end

# Convenience: allocate given dim for a fixed-level Tensor{T,M}
function Tensor{T,M}(dim::Int) where {T,M}
    offsets = level_starts0(dim, M)
    coeffs  = Vector{T}(undef, offsets[end])
    return Tensor{T,M}(coeffs, dim, offsets)
end

# -------------------------------------------------------------------
# Dense ↔ Dense (respects per-level padding)
# -------------------------------------------------------------------

function Base.isapprox(a::Chen.Tensor{Ta,Ma}, b::Chen.Tensor{Tb,Mb};
                       atol::Real=1e-8, rtol::Real=1e-8) where {Ta,Tb,Ma,Mb}
    a.dim == b.dim && Ma == Mb || return false
    sA, sB = a.offsets, b.offsets
    d, m = a.dim, Ma
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

# -------------------------------------------------------------------
# Tensor interface
# -------------------------------------------------------------------

Base.length(ts::Tensor{T,M}) where {T,M} = length(ts.coeffs)
Base.getindex(ts::Tensor{T,M}, i::Int) where {T,M} = ts.coeffs[i]
Base.eltype(::Tensor{T,M}) where {T,M} = T

Base.show(io::IO, ts::Tensor{T,M}) where {T,M} =
    print(io, "Tensor(dim=$(ts.dim), level=$(M), length=$(length(ts)))")

# Allocate a new Tensor with the same "shape" (dim, level)
Base.similar(ts::Tensor{T,M}) where {T,M} = Tensor{T,M}(ts.dim)

# Same, but change element type
Base.similar(ts::Tensor{T,M}, ::Type{S}) where {T,M,S} = Tensor{S,M}(ts.dim)

# Allocate-and-copy convenience
Base.copy(ts::Tensor{T,M}) where {T,M} =
    Tensor{T,M}(copy(ts.coeffs), ts.dim, ts.offsets)

# In-place copy with shape check
function Base.copy!(dest::Tensor{T,M}, src::Tensor{T,M}) where {T,M}
    @assert dest.dim == src.dim "Tensor shape mismatch"
    copyto!(dest.coeffs, src.coeffs)
    return dest
end

dim(ts::Tensor)   = ts.dim
level(ts::Tensor{T,M}) where {T,M} = M
coeffs(ts::Tensor) = ts.coeffs
offsets(ts::Tensor) = ts.offsets

# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

@inline function _zero!(ts::Tensor{T,M}) where {T,M}
    fill!(ts.coeffs, zero(T)); ts
end

@inline function add_scaled!(dest::Tensor{T,M}, src::Tensor{T,M}, α::T) where {T,M}
    @inbounds @simd for i in eachindex(dest.coeffs, src.coeffs)
        dest.coeffs[i] = muladd(α, src.coeffs[i], dest.coeffs[i])
    end
    dest
end

@inline _write_unit!(t::Tensor{T,M}) where {T,M} =
    (t.coeffs[t.offsets[1] + 1] = one(T); t)

function lift1!(ts::Tensor{T,M}, x::AbstractVector{T}) where {T,M}
    @assert length(x) == ts.dim
    fill!(ts.coeffs, zero(T))
    @inbounds copyto!(ts.coeffs, 1, x, 1, ts.dim)  # write level-1 block
    return ts
end

# -------------------------------------------------------------------
# exp!(out, x)
# -------------------------------------------------------------------

"""
    exp!(out, x::AbstractVector)

Compute the truncated tensor exponential of a level-1 element `x`:
    exp(x) = x + x^2/2! + ... + x^m/m!
(level-0 unit is stored at level-0 in `out`). Works for any Tensor backend.
"""
function exp!(out::Tensor{T,M}, x::AbstractVector{T}) where {T,M}
    d = out.dim
    m = M
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

@generated function mul_grouplike!(
    out_tensor::Tensor{T,M}, x1_tensor::Tensor{T,M}, x2_tensor::Tensor{T,M}
) where {T,M}
    # Build per-level blocks at compile time
    level_blocks = Expr[]
    for k in 1:M
        push!(level_blocks, quote
            # ---- level $k ----
            out_start = offsets[$k + 1] + 1

            # Base term: out_k = x1_k + x2_k
            @avx for j in 0:out_len-1
                out[out_start + j] = x1[out_start + j] + x2[out_start + j]
            end

            # Middle terms: i = 1..$(k-1)
            a_len = d
            for i in 1:$(k-1)
                a_block_start = offsets[i + 1]
                b_block_start = offsets[$k - i + 1]
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

            # Prepare for next level
            out_len *= d
        end)
    end

    return quote
        out = out_tensor.coeffs
        x1  = x1_tensor.coeffs
        x2  = x2_tensor.coeffs

        d       = out_tensor.dim
        offsets = out_tensor.offsets

        # level-0 index
        i0 = offsets[1] + 1

        # Assume group-like in hot path; uncomment for debug checks:
        # @assert x1[i0] == one(T) "mul_grouplike!: expected x1 level-0 == 1"
        # @assert x2[i0] == one(T) "mul_grouplike!: expected x2 level-0 == 1"

        # level-0: 1 * 1 = 1
        out[i0] = one(T)

        # length of level-1 block
        out_len = d

        @inbounds begin
            $(Expr(:block, level_blocks...))
        end

        out_tensor
    end
end

# -------------------------------------------------------------------
# mul!: generic path (arbitrary level-0 coefficients a0, b0)
# -------------------------------------------------------------------

@inline function mul!(
    out_tensor::Tensor{T,M}, x1_tensor::Tensor{T,M}, x2_tensor::Tensor{T,M}
) where {T,M}
    out, x1, x2, m = out_tensor.coeffs, x1_tensor.coeffs, x2_tensor.coeffs, M
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

# -------------------------------------------------------------------
# Internal helper: propagate segment offsets between tensor levels
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# level_starts0: offsets (0-based) for each level block
# -------------------------------------------------------------------

"""
    level_starts0(dim::Int, level::Int) -> Vector{Int}

Return a vector `s` of length `level + 2` with **0-based** start indices for the
flattened tensor-series blocks of each level `k = 1..level` in column-major layout,
plus a final sentinel.

Invariant:
- `s[1] == 0`
- `s[k+1] - s[k] == dim^k` (length of level-`k` block)
- `s[level+1] == (dim^(level+1) - dim) ÷ (dim - 1)` (total length)
"""
function level_starts0(d::Int, m::Int)
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

# -------------------------------------------------------------------
# log! and log
# -------------------------------------------------------------------

"""
    log!(out::Tensor{T,M}, g::Tensor{T,M}) where T

Compute the truncated tensor logarithm of a group-like element `g` up to `out.level`,
using the power-series:
    log(g) = (g-1) - (g-1)^2/2 + (g-1)^3/3 - ...  (truncated at level M)
All multiplications are done with `mul!`. The result has zero level-0.
"""
function log!(out::Tensor{T,M}, g::Tensor{T,M}) where {T,M}
    @assert out.dim == g.dim "log!: shape mismatch"
    m = M
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
    log(g::Tensor{T,M}) -> Tensor{T,M}

Allocating wrapper for `log!`.
"""
function log(g::Tensor{T,M}) where {T,M}
    out = similar(g)
    return log!(out, g)
end
