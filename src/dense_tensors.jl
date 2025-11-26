using StaticArrays
using LoopVectorization: @avx, @turbo

# -------------------------------------------------------------------
# Tensor type: specialize on Level M AND Dimension D
# -------------------------------------------------------------------

struct Tensor{T,D,M} <: AbstractTensor{T}
    coeffs::Vector{T}
    offsets::Vector{Int}
end

# Accessors
dim(::Tensor{T,D,M}) where {T,D,M} = D
level(::Tensor{T,D,M}) where {T,D,M} = M
Base.parent(ts::Tensor) = ts.coeffs
coeffs(ts::Tensor) = ts.coeffs
offsets(ts::Tensor) = ts.offsets

Base.eltype(::Tensor{T,D,M}) where {T,D,M} = T
Base.length(ts::Tensor) = length(ts.coeffs)
@inline Base.getindex(ts::Tensor, i::Int) = @inbounds ts.coeffs[i]
@inline Base.setindex!(ts::Tensor, v, i::Int) = @inbounds (ts.coeffs[i] = v)

Base.show(io::IO, ts::Tensor{T,D,M}) where {T,D,M} =
    print(io, "Tensor{T=$T, D=$D, M=$M}(length=$(length(ts.coeffs)))")

# -------------------------------------------------------------------
# Constructors
# -------------------------------------------------------------------

function Tensor{T,D,M}() where {T,D,M}
    offsets = level_starts0(D, M)
    coeffs  = Vector{T}(undef, offsets[end])
    return Tensor{T,D,M}(coeffs, offsets)
end

function Tensor{T,D,M}(coeffs::Vector{T}) where {T,D,M}
    offsets = level_starts0(D, M)
    @assert length(coeffs) == offsets[end] "Coefficient length mismatch"
    return Tensor{T,D,M}(coeffs, offsets)
end

# Dynamic Factory
function Tensor(coeffs::Vector{T}, d::Int, m::Int) where {T}
    return _make_tensor(coeffs, Val(d), Val(m))
end

@generated function _make_tensor(coeffs::Vector{T}, ::Val{D}, ::Val{M}) where {T,D,M}
    quote
        return Tensor{T,D,M}(coeffs)
    end
end

Base.similar(ts::Tensor{T,D,M}) where {T,D,M} = Tensor{T,D,M}()
Base.similar(ts::Tensor{T,D,M}, ::Type{S}) where {T,D,M,S} = Tensor{S,D,M}()

Base.copy(ts::Tensor{T,D,M}) where {T,D,M} = 
    Tensor{T,D,M}(copy(ts.coeffs), ts.offsets)

function Base.copy!(dest::Tensor{T,D,M}, src::Tensor{T,D,M}) where {T,D,M}
    copyto!(dest.coeffs, src.coeffs)
    return dest
end

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

function level_starts0(d::Int, m::Int)
    offsets = Vector{Int}(undef, m + 2)
    offsets[1] = 0
    len = 1
    @inbounds for k in 1:m+1
        offsets[k+1] = offsets[k] + len
        len *= d
    end
    # Padding for alignment (64-byte / 8 doubles)
    W = 8 
    pad = (W - (offsets[2] % W)) % W
    if pad != 0
        @inbounds for k in 2:length(offsets)
            offsets[k] += pad
        end
    end
    return offsets
end

@inline function _zero!(ts::Tensor{T}) where {T}
    fill!(ts.coeffs, zero(T)); ts
end

@inline _write_unit!(t::Tensor{T}) where {T} =
    (t.coeffs[t.offsets[1] + 1] = one(T); t)

# -------------------------------------------------------------------
# exp! (Specialized)
# -------------------------------------------------------------------

@generated function exp!(out::Tensor{T,D,M}, x::SVector{D,T}) where {T,D,M}
    off = level_starts0(D, M)
    
    level_loops = Expr[]
    
    for k in 2:M
        prev_len_val = D^(k-1)
        prev_start = off[k] + 1
        cur_start  = off[k+1] + 1
        
        # Unroll the loop over D dimensions
        push!(level_loops, quote
            scale = inv(T($k))
            Base.@nexprs $D i -> begin
                val = scale * x[i]
                dest_ptr = $cur_start + (i - 1) * $prev_len_val
                # Vectorized copy-scale
                @turbo for j in 0:$(prev_len_val - 1)
                    coeffs[dest_ptr + j] = val * coeffs[$prev_start + j]
                end
            end
        end)
    end

    quote
        coeffs = out.coeffs
        # Level 0
        @inbounds coeffs[$(off[1] + 1)] = one(T)
        
        # Level 1
        start1 = $(off[2] + 1)
        @inbounds for i in 1:D
            coeffs[start1 + i - 1] = x[i]
        end
        
        # Levels 2..M
        @inbounds begin
            $(Expr(:block, level_loops...))
        end
        return nothing
    end
end

# -------------------------------------------------------------------
# mul_accumulate! (Optimized)
# -------------------------------------------------------------------

"""
    mul_accumulate!(S, seg)
Updates S = S ⊗ seg in-place.
Iterates levels downwards.
"""
@generated function mul_accumulate!(
    S_tensor::Tensor{T,D,M}, seg_tensor::Tensor{T,D,M}
) where {T,D,M}
    
    off = level_starts0(D, M)
    block_updates = Expr[]
    
    for k in M:-1:1
        updates_k = Expr[]
        
        out_start = off[k+1] + 1
        len_k = D^k
        
        # 1. Base term: S^k += seg^k (since S^0 = 1)
        push!(updates_k, quote
             @turbo for j in 0:$(len_k - 1)
                 S[$(out_start) + j] += seg[$(out_start) + j]
             end
        end)
        
        # 2. Convolution: S^k += S^{k-j} ⊗ seg^j
        for j in 1:(k-1)
            s_prev_start = off[k - j + 1] + 1
            e_j_start    = off[j + 1] + 1
            
            dim_s = D^(k-j) 
            dim_e = D^j     

            # We fuse the inner loop logic. 
            # Note: For small D^j, @turbo is good.
            push!(updates_k, quote
                for u in 0:$(dim_s - 1)
                    val_s = S[$s_prev_start + u]
                    row_target = $out_start + u * $dim_e
                    @turbo for v in 0:$(dim_e - 1)
                        S[row_target + v] += val_s * seg[$e_j_start + v]
                    end
                end
            end)
        end
        
        push!(block_updates, Expr(:block, updates_k...))
    end

    return quote
        S   = S_tensor.coeffs
        seg = seg_tensor.coeffs
        
        @inbounds begin
            $(Expr(:block, block_updates...))
        end
        return S_tensor
    end
end

# -------------------------------------------------------------------
# mul! (Generic)
# -------------------------------------------------------------------

@generated function mul!(
    out_tensor::Tensor{T,D,M}, x1_tensor::Tensor{T,D,M}, x2_tensor::Tensor{T,D,M}
) where {T,D,M}
    off = level_starts0(D, M)
    level_blocks = Expr[]
    
    for k in 1:M
        out_len_k = D^k
        out_start = off[k+1] + 1
        
        push!(level_blocks, quote
            if a0 == one(T)
                copyto!(out, $out_start, x2, $out_start, $out_len_k)
            elseif a0 == zero(T)
                @turbo for j in 0:$(out_len_k-1); out[$out_start + j] = zero(T); end
            else
                @turbo for j in 0:$(out_len_k-1); out[$out_start + j] = a0 * x2[$out_start + j]; end
            end
            
            $(let inner = Expr[]
                a_len = D
                for i in 1:(k-1)
                    b_len = D^(k-i)
                    a_start = off[i+1] + 1
                    b_start = off[k-i+1] + 1
                    push!(inner, quote
                        for ai in 0:$(a_len-1)
                            val_a = x1[$a_start + ai]
                            row_t = $out_start + ai * $b_len
                            @turbo for bi in 0:$(b_len-1)
                                out[row_t + bi] += val_a * x2[$b_start + bi]
                            end
                        end
                    end)
                    a_len *= D
                end
                Expr(:block, inner...)
            end)

            if b0 != zero(T)
                if b0 == one(T)
                    @turbo for j in 0:$(out_len_k-1); out[$out_start + j] += x1[$out_start + j]; end
                else
                    @turbo for j in 0:$(out_len_k-1); out[$out_start + j] += b0 * x1[$out_start + j]; end
                end
            end
        end)
    end

    quote
        out = out_tensor.coeffs
        x1  = x1_tensor.coeffs
        x2  = x2_tensor.coeffs
        a0 = x1[$(off[1]+1)]; b0 = x2[$(off[1]+1)]
        out[$(off[1]+1)] = a0 * b0
        @inbounds begin
            $(Expr(:block, level_blocks...))
        end
        return out_tensor
    end
end

# -------------------------------------------------------------------
# Log
# -------------------------------------------------------------------

function log!(out::Tensor{T,D,M}, g::Tensor{T,D,M}) where {T,D,M}
    offsets = out.offsets
    i0 = offsets[1] + 1
    
    X = similar(out)
    copy!(X, g)
    X.coeffs[i0] -= one(T) 

    _zero!(out)
    P = similar(out) 
    Q = similar(out) 
    copy!(P, X)

    sgn = one(T)
    for k in 1:M
        add_scaled!(out, P, sgn / T(k))
        if k < M
            mul!(Q, P, X)
            Q.coeffs[i0] = zero(T)
            P, Q = Q, P
        end
        sgn = -sgn
    end
    out.coeffs[i0] = zero(T)
    return out
end

function log(g::Tensor{T,D,M}) where {T,D,M}
    out = similar(g)
    return log!(out, g)
end

@inline function add_scaled!(dest::Tensor{T,D,M}, src::Tensor{T,D,M}, α::T) where {T,D,M}
    @inbounds @turbo for i in eachindex(dest.coeffs, src.coeffs)
        dest.coeffs[i] = muladd(α, src.coeffs[i], dest.coeffs[i])
    end
    dest
end