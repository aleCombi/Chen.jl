module PathSignatures

using StaticArrays
using LoopVectorization: @avx

export TensorSeries, signature_path, signature_words, all_signature_words

# ---------------- types ----------------

struct TensorSeries{T}
    coeffs::Vector{T}   # flat coefficients in tensor basis
    dim::Int            # ambient dimension
    level::Int          # truncation level
end

Base.length(ts::TensorSeries) = length(ts.coeffs)
Base.getindex(ts::TensorSeries, i::Int) = ts.coeffs[i]
Base.show(io::IO, ts::TensorSeries) =
    print(io, "TensorSeries(dim=$(ts.dim), level=$(ts.level), length=$(length(ts)))")

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
    d = length(Δ)
    @turbo for i in 1:d, j in 1:prev_len
        s = scale * Δ[i]
        base = cur_start + (i - 1) * prev_len - 1
        out[base + j] = s * out[prev_start + j - 1]
    end
    return nothing
end

# ---- overload for *AbstractVector* endpoints ----
@inline function segment_signature!(
    out::StridedVector{T}, a::AbstractVector{T}, b::AbstractVector{T}, m::Int,
    buffer::StridedVector{T}, inv_level::AbstractVector{T}
) where {T}
    d = length(a)
    @assert length(b) == d
    @assert length(buffer) >= d

    @inbounds begin
        # Δ = b - a
        @simd for i in 1:d
            buffer[i] = b[i] - a[i]
        end

        # level 1
        idx    = 1
        curlen = d
        copyto!(out, idx, buffer, 1, d)
        prev_start = idx
        idx += curlen

        # quick return
        if m == 1
            @assert idx - 1 == length(out)
            return nothing
        end

        # levels 2..m
        for level in 2:m
            prev_len  = curlen
            curlen   *= d
            cur_start = idx
            _segment_level_offsets!(out, buffer, inv_level[level],
                                    prev_start, prev_len, cur_start)
            prev_start = cur_start
            idx += curlen
        end
    end

    # cheaper postcondition: avoids pow/div
    @assert idx - 1 == length(out)
    return nothing
end

@inline function chen_product!(
    out::StridedVector{T}, x1::StridedVector{T}, x2::StridedVector{T},
    m::Int, offsets::Vector{Int}
) where {T}
    @inbounds for k in 1:m
        out_start = offsets[k] + 1
        out_len   = offsets[k+1] - offsets[k]

        # ---- init with i = 0 term: out_k = 1 ⊗ x2_k = x2_k (full-block copy)
        b_start = offsets[k]                  # 0-based in offsets
        copyto!(out, out_start, x2, b_start + 1, out_len)

        # ---- middle terms: i = 1 .. k-1  (outer products, +=)
        for i in 1:(k-1)
            a_start = offsets[i]
            a_len   = offsets[i+1] - offsets[i]
            b_start = offsets[k - i]
            b_len   = offsets[k - i + 1] - offsets[k - i]

            @avx for ai in 1:a_len, bi in 1:b_len
                row0 = out_start + (ai - 1) * b_len - 1
                out[row0 + bi] = muladd(x1[a_start + ai], x2[b_start + bi], out[row0 + bi])
            end
        end

        # ---- endpoint term: i = k  (x1_k ⊗ 1) → contiguous add
        # This is a_len = out_len, b_len = 1 → flatten to one loop.
        a_start = offsets[k]
        @avx for j in 1:out_len
            out[out_start + j - 1] += x1[a_start + j]
        end
    end
    return out
end


# ---------------- public API ----------------

function signature_path(path::Vector{SVector{D,T}}, m::Int) where {D,T}
    d = D
    total_terms = div(d^(m + 1) - d, d - 1)

    offsets = Vector{Int}(undef, m + 1)
    offsets[1] = 0
    len = d
    for k in 1:m
        offsets[k+1] = offsets[k] + len
        len *= d
    end

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
        chen_product!(b, a, segment, m, offsets)
        a, b = b, a
    end

    return TensorSeries(a, d, m)
end

include("tensor_algebra.jl")
include("vol_signature.jl")
include("tensor_conversions.jl")

end # module PathSignatures
