module PathSignatures

using StaticArrays, LoopVectorization

export signature_path, signature_words, all_signature_words

function signature_words(level::Int, dim::Int)
    Iterators.product(ntuple(_ -> 1:dim, level)...)
end

function all_signature_words(max_level::Int, dim::Int)
    Iterators.flatten(signature_words(ℓ, dim) for ℓ in 1:max_level)
end

# ---------------- internals ----------------

@inline function _segment_level_offsets!(
    out::StridedVector{T}, Δ::StridedVector{T}, m::Int,
    prev_start::Int, prev_len::Int, cur_start::Int
) where {T}
    scale = inv(T(m))
    @inbounds for i in 1:length(Δ)
        s = scale * Δ[i]
        base = cur_start + (i - 1) * prev_len
        @simd for j in 1:prev_len
            out[base + j - 1] = s * out[prev_start + j - 1]
        end
    end
    return nothing
end

# ---- overload for *AbstractVector* endpoints (fixes your MethodError) ----
function segment_signature!(
    out::StridedVector{T}, a::AbstractVector{T}, b::AbstractVector{T}, m::Int,
    buffer::StridedVector{T}
) where {T}
    d = length(a)
    @assert length(b) == d
    @assert length(buffer) >= d

    @inbounds @simd for i in 1:d
        buffer[i] = b[i] - a[i]
    end
    displacement = @view buffer[1:d]

    @assert length(out) == div(d^(m + 1) - d, d - 1)

    idx = 1
    curlen = d
    @inbounds copyto!(out, idx, displacement, 1, curlen)
    prev_start = idx
    idx += curlen

    for level in 2:m
        prev_len = curlen
        curlen *= d
        cur_start = idx
        _segment_level_offsets!(out, displacement, level, prev_start, prev_len, cur_start)
        prev_start = cur_start
        idx += curlen
    end
    return nothing
end

# ---- SVector overload (fast path when you have SVectors) ----
function segment_signature!(
    out::StridedVector{T}, a::SVector{D,T}, b::SVector{D,T}, m::Int, buffer::StridedVector{T}
) where {D,T}
    d = D
    @assert length(buffer) >= d
    @inbounds @simd for i in 1:d
        buffer[i] = b[i] - a[i]
    end
    displacement = @view buffer[1:d]

    @assert length(out) == div(d^(m + 1) - d, d - 1)

    idx = 1
    curlen = d
    @inbounds copyto!(out, idx, displacement, 1, curlen)
    prev_start = idx
    idx += curlen

    for level in 2:m
        prev_len = curlen
        curlen *= d
        cur_start = idx
        _segment_level_offsets!(out, displacement, level, prev_start, prev_len, cur_start)
        prev_start = cur_start
        idx += curlen
    end
    return nothing
end

@inline function _zero_range!(x::StridedVector{T}, start::Int, len::Int) where {T}
    @inbounds @simd for i in 0:len-1
        x[start + i] = zero(T)
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

            @simd for ai in 1:a_len
                a_val = (i == 0) ? one(T) : x1[a_start + ai]
                row_base = out_start + (ai - 1) * b_len
                @simd for bi in 1:b_len
                    b_val = (k == i) ? one(T) : x2[b_start + bi]
                    out[row_base + bi - 1] += a_val * b_val
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

    offsets = Vector{Int}(undef, m + 1)
    offsets[1] = 0
    len = d
    for k in 1:m
        offsets[k+1] = offsets[k] + len
        len *= d
    end

    a = Vector{T}(undef, total_terms)
    b = Vector{T}(undef, total_terms)
    segment = Vector{T}(undef, total_terms)
    dispbuf = Vector{T}(undef, d)

    segment_signature!(a, path[1], path[2], m, dispbuf)

    for i in 2:length(path)-1
        segment_signature!(segment, path[i], path[i+1], m, dispbuf)
        chen_product!(b, a, segment, d, m, offsets)
        a, b = b, a
    end
    return a
end


include("tensor_algebra.jl") 
include("vol_signature.jl")
include("tensor_conversions.jl")

end # module PathSignatures
