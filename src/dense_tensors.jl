abstract type AbstractTensor{T} end

struct Tensor{T} <: AbstractTensor{T}
    coeffs::StridedVector{T}
    dim::Int
    level::Int
    offsets::Vector{Int}
    
    function Tensor(coeffs::StridedVector{T}, dim::Int, level::Int) where {T}
        offsets = level_starts0(dim, level)
        new{T}(coeffs, dim, level, offsets)
    end

    function Tensor{T}(dim::Int, level::Int) where {T}
        offsets = level_starts0(dim, level)     # includes pad + end sentinel
        coeffs  = Vector{T}(undef, offsets[end])
        new{T}(coeffs, dim, level, offsets)
    end
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

function power_elevate!(out::AT, t::AT, n::Int) where {AT<:AbstractTensor}
    n == 0 && return (exp!(out, zeros(eltype(t), t.dim)); out)  # identity
    n == 1 && return copyto!(out, t)

    copyto!(out, t)
    tmp = similar(out)
    for k in 2:n
        mul!(tmp, out, t)
        out, tmp = tmp, out
    end
    return out
end

# --- small helpers ------------------------------------------------------------

@inline function _zero!(ts::Tensor{T}) where {T}
    fill!(ts.coeffs, zero(T)); ts
end

@inline function _add_scaled!(dest::Tensor{T}, src::Tensor{T}, α::T) where {T}
    @inbounds @simd for i in eachindex(dest.coeffs, src.coeffs)
        dest.coeffs[i] = muladd(α, src.coeffs[i], dest.coeffs[i])
    end
    dest
end

"""
    exp!(out, X)

Compute the truncated power-series exponential:
    out = X + X^2/2! + ... + X^m/m!
(Level-0 unit is implicit and not stored; backends control truncation via `level`.)
Works for any `Tensor` backend implementing the small interface.
"""
function exp!(out::AbstractTensor{T}, X::AbstractTensor{T}) where {T}
    @assert dim(out)   == dim(X)
    @assert level(out) == level(X)

    _zero!(out)
    m = level(X)
    m == 0 && return out

    # term = X^1
    term = similar(X)
    copy!(term, X)

    invfact = one(T)                # 1/1!
    add_scaled!(out, term, invfact) # add X

    # tmp buffer for powers
    tmp = similar(X)

    @inbounds for k in 2:m
        # tmp <- term * X   (do not assume alias-safety)
        mul!(tmp, term, X)
        term, tmp = tmp, term

        invfact *= inv(T(k))        # update 1/k!
        add_scaled!(out, term, invfact)
    end
    return out
end

# Allocating wrapper (works for any backend)
function exp(X::AT) where {AT<:AbstractTensor}
    out = similar(X)
    exp!(out, X)
    return out
end

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
function exp!(out::AT, x::AbstractVector{T}) where {T,AT<:AbstractTensor{T}}
    # Build the level-1 tensor X from x
    X = similar(out)::AT
    lift1!(X, x)

    # out ← exp(X) via the generic tensor power-series kernel
    return exp!(out, X)
end



function mul(a::AbstractTensor, b::AbstractTensor)
    # allocate same "shape" as a, using eltype promotion
    dest = similar(a, promote_type(eltype(a), eltype(b)))
    return mul!(dest, a, b)
end

⊗(a::AbstractTensor, b::AbstractTensor) = mul(a, b)

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
    out::StridedVector{T}, Δ::StridedVector{T}, scale::T,
    prev_start::Int, prev_len::Int, cur_start::Int
) where {T}
    d = length(Δ)
    @inbounds @avx for i in 1:d, j in 1:prev_len
        s = scale * Δ[i]
        base = cur_start + (i - 1) * prev_len - 1
        out[base + j] = s * out[prev_start + j - 1]
    end
    return nothing
end

@inline function exp!(
    out::Tensor{T}, x::StridedVector{T}
) where {T}
    @assert length(x) == out.dim "exp!: length(x)=$(length(x)) must equal dim=$(out.dim)"

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
        @assert idx - 1 == length(out)
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

    # @assert idx - 1 == length(out)
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

Example:
```julia
julia> level_starts0(2, 3)
4-element Vector{Int}: [0, 2, 6, 14]   # sizes 2,4,8 → cumulative 0,2,6,14
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

