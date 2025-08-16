module PathSignatures

using StaticArrays
using LoopVectorization: @avx

export signature_path, signature_words, all_signature_words

# ---------------- utilities ----------------

function signature_words(level::Int, dim::Int)
    Iterators.product(ntuple(_ -> 1:dim, level)...)
end

function all_signature_words(max_level::Int, dim::Int)
    Iterators.flatten(signature_words(ℓ, dim) for ℓ in 1:max_level)
end

# ---------------- internals ----------------

@inline function _segment_level_offsets!(
    out::StridedVector{T}, Δ::StridedVector{T}, scale::T,
    prev_start::Int, prev_len::Int, cur_start::Int
) where {T}
    @inbounds for i in 1:length(Δ)
        s = scale * Δ[i]
        base = cur_start + (i - 1) * prev_len
        # long inner loop → AVX helps
        @avx for j in 1:prev_len
            out[base + j - 1] = s * out[prev_start + j - 1]
        end
    end
    return nothing
end

# zero a contiguous slice without creating a view
@inline function _zero_range!(x::StridedVector{T}, start::Int, len::Int) where {T}
    @inbounds @simd for u in 1:len
        x[start + u - 1] = zero(T)
    end
    return nothing
end

# ---- overload for *AbstractVector* endpoints ----
function segment_signature!(
    out::StridedVector{T}, a::AbstractVector{T}, b::AbstractVector{T}, m::Int,
    buffer::StridedVector{T}, inv_level::AbstractVector{T}
) where {T}
    d = length(a)
    @assert length(b) == d
    @assert length(buffer) >= d

    # displacement in buffer[1:d]
    @inbounds @simd for i in 1:d
        buffer[i] = b[i] - a[i]
    end

    @assert length(out) == div(d^(m + 1) - d, d - 1)

    # level 1
    idx = 1
    curlen = d
    @inbounds copyto!(out, idx, buffer, 1, d)
    prev_start = idx
    idx += curlen

    # levels 2..m
    for level in 2:m
        prev_len  = curlen
        curlen   *= d
        cur_start = idx
        _segment_level_offsets!(out, buffer, inv_level[level], prev_start, prev_len, cur_start)
        prev_start = cur_start
        idx += curlen
    end
    return nothing
end

# ---- SVector overload (fast path) ----
function segment_signature!(
    out::StridedVector{T}, a::SVector{D,T}, b::SVector{D,T}, m::Int,
    buffer::StridedVector{T}, inv_level::AbstractVector{T}
) where {D,T}
    d = D
    @assert length(buffer) >= d

    @inbounds @simd for i in 1:d
        buffer[i] = b[i] - a[i]
    end

    @assert length(out) == div(d^(m + 1) - d, d - 1)

    idx = 1
    curlen = d
    @inbounds copyto!(out, idx, buffer, 1, d)
    prev_start = idx
    idx += curlen

    for level in 2:m
        prev_len  = curlen
        curlen   *= d
        cur_start = idx
        _segment_level_offsets!(out, buffer, inv_level[level], prev_start, prev_len, cur_start)
        prev_start = cur_start
        idx += curlen
    end
    return nothing
end

@inline function chen_product!(
    out::StridedVector{T}, x1::StridedVector{T}, x2::StridedVector{T},
    d::Int, m::Int, offsets::Vector{Int}
) where {T}
    @inbounds for k in 1:m
        out_start = offsets[k] + 1
        out_len   = offsets[k+1] - offsets[k]
        _zero_range!(out, out_start, out_len)

        for i in 0:k
            a_start = (i == 0) ? 0 : offsets[i]
            a_len   = (i == 0) ? 1 : (offsets[i+1] - offsets[i])
            b_start = (k == i) ? 0 : offsets[k-i]
            b_len   = (k == i) ? 1 : (offsets[k-i+1] - offsets[k-i])

            if i == 0 && k == i
                # both sides are 1 → add ones
                @inbounds @simd for ai in 1:a_len
                    row_base = out_start + (ai - 1) * b_len
                    @inbounds @simd for bi in 1:b_len
                        out[row_base + bi - 1] += one(T)
                    end
                end
            elseif i == 0
                # left is 1, right is x2
                @inbounds for ai in 1:a_len
                    row_base = out_start + (ai - 1) * b_len
                    @inbounds @simd for bi in 1:b_len
                        out[row_base + bi - 1] += x2[b_start + bi]
                    end
                end
            elseif k == i
                # right is 1, left is x1
                @inbounds for ai in 1:a_len
                    a_val = x1[a_start + ai]
                    row_base = out_start + (ai - 1) * b_len
                    @inbounds @simd for bi in 1:b_len
                        out[row_base + bi - 1] += a_val
                    end
                end
            else
                # general case: outer product
                @inbounds for ai in 1:a_len
                    a_val = x1[a_start + ai]
                    row_base = out_start + (ai - 1) * b_len
                    @inbounds @simd for bi in 1:b_len
                        out[row_base + bi - 1] += a_val * x2[b_start + bi]
                    end
                end
            end
        end
    end
    return out
end

# ---------------- public API ----------------

function signature_path(path::Vector{SVector{D,T}}, m::Int) where {D,T}
    d = D
    total_terms = div(d^(m + 1) - d, d - 1)

    # precompute offsets (no ^ in loop)
    offsets = Vector{Int}(undef, m + 1)
    offsets[1] = 0
    len = d
    for k in 1:m
        offsets[k+1] = offsets[k] + len
        len *= d
    end

    # precompute inverse levels
    inv_level = Vector{T}(undef, m)
    @inbounds for k in 1:m
        inv_level[k] = inv(T(k))
    end

    a       = Vector{T}(undef, total_terms)
    b       = Vector{T}(undef, total_terms)
    segment = Vector{T}(undef, total_terms)
    dispbuf = Vector{T}(undef, d)

    segment_signature!(a, path[1], path[2], m, dispbuf, inv_level)

    for i in 2:length(path)-1
        segment_signature!(segment, path[i], path[i+1], m, dispbuf, inv_level)
        chen_product!(b, a, segment, d, m, offsets)
        a, b = b, a
    end
    return a
end

include("tensor_algebra.jl")
include("vol_signature.jl")
include("tensor_conversions.jl")

end # module PathSignatures
