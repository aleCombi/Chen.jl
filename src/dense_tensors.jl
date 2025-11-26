using StaticArrays
using LoopVectorization: @avx, @turbo

# -------------------------------------------------------------------
# Tensor type: specialize on Level M AND Dimension D
# -------------------------------------------------------------------

struct Tensor{T,D,M} <: AbstractTensor{T}
    coeffs::Vector{T}
    # offsets are kept for runtime generic access, but 
    # optimized kernels calculate them as constants.
    offsets::Vector{Int}
end

# Accessors
dim(::Tensor{T,D,M}) where {T,D,M} = D
level(::Tensor{T,D,M}) where {T,D,M} = M
coeffs(ts::Tensor) = ts.coeffs
offsets(ts::Tensor) = ts.offsets

Base.eltype(::Tensor{T,D,M}) where {T,D,M} = T
Base.length(ts::Tensor) = length(ts.coeffs)
Base.getindex(ts::Tensor, i::Int) = ts.coeffs[i]
Base.setindex!(ts::Tensor, v, i::Int) = (ts.coeffs[i] = v)

Base.show(io::IO, ts::Tensor{T,D,M}) where {T,D,M} =
    print(io, "Tensor{T=$T, D=$D, M=$M}(length=$(length(ts.coeffs)))")

# -------------------------------------------------------------------
# Constructors
# -------------------------------------------------------------------

# 1. Static Constructor (Allocating)
function Tensor{T,D,M}() where {T,D,M}
    offsets = level_starts0(D, M)
    coeffs  = Vector{T}(undef, offsets[end])
    return Tensor{T,D,M}(coeffs, offsets)
end

# 2. Internal Constructor (wrapping existing vector)
function Tensor{T,D,M}(coeffs::Vector{T}) where {T,D,M}
    offsets = level_starts0(D, M)
    @assert length(coeffs) == offsets[end] "Coefficient length mismatch"
    return Tensor{T,D,M}(coeffs, offsets)
end

# 3. Dynamic Factory (Value-to-Type Bridge)
function Tensor(coeffs::Vector{T}, d::Int, m::Int) where {T}
    return _make_tensor(coeffs, Val(d), Val(m))
end

@generated function _make_tensor(coeffs::Vector{T}, ::Val{D}, ::Val{M}) where {T,D,M}
    quote
        return Tensor{T,D,M}(coeffs)
    end
end

# 4. Similar / Copy
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

"""
    level_starts0(dim, m)
Calculates 0-based offsets for each level block, including padding for alignment.
"""
function level_starts0(d::Int, m::Int)
    offsets = Vector{Int}(undef, m + 2)
    offsets[1] = 0
    len = 1
    @inbounds for k in 1:m+1
        offsets[k+1] = offsets[k] + len
        len *= d
    end
    # Ensure 64-byte alignment for the start of level 1 (offsets[2])
    # This helps SIMD operations on the vector part.
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
    # Pre-calculate offsets during compilation to bake them in as constants
    off = level_starts0(D, M)
    
    level_loops = Expr[]
    
    for k in 2:M
        prev_len_val = D^(k-1)
        prev_start = off[k] + 1
        cur_start  = off[k+1] + 1
        
        # Generation of the loop for level k
        push!(level_loops, quote
            scale = inv(T($k))
            for i in 1:D
                val = scale * x[i]
                # Pointers relative to the start of the coefficient vector
                dest_ptr = $cur_start + (i - 1) * $prev_len_val
                
                # SIMD loop
                @turbo for j in 0:$(prev_len_val - 1)
                    coeffs[dest_ptr + j] = val * coeffs[$prev_start + j]
                end
            end
        end)
    end

    quote
        coeffs = out.coeffs
        
        # Level 0: 1.0
        coeffs[$(off[1] + 1)] = one(T)
        
        # Level 1: copy vector x
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
# mul_accumulate! (The Speed Booster)
# -------------------------------------------------------------------

"""
    mul_accumulate!(S::Tensor{T,D,M}, seg::Tensor{T,D,M})

Updates `S` in-place via `S = S ⊗ seg`.
Optimized for path signatures where `S` accumulates segments.
Iterates backwards from M to 1 to allow single-buffer updates.
Assumes S[0]==1 and seg[0]==1.
"""
@generated function mul_accumulate!(
    S_tensor::Tensor{T,D,M}, seg_tensor::Tensor{T,D,M}
) where {T,D,M}
    
    off = level_starts0(D, M)
    block_updates = Expr[]
    
    # Iterate backwards from level k = M down to 1
    for k in M:-1:1
        updates_k = Expr[]
        
        out_start = off[k+1] + 1
        len_k = D^k
        
        # 1. Base term: S^k += seg^k (Coming from S^0 ⊗ seg^k)
        push!(updates_k, quote
             @turbo for j in 0:$(len_k - 1)
                 S[$(out_start) + j] += seg[$(out_start) + j]
             end
        end)
        
        # 2. Convolution terms: S^k += S^{k-j} ⊗ seg^j
        for j in 1:(k-1)
            # S^{k-j} index
            s_prev_start = off[k - j + 1] + 1
            # seg^j index
            e_j_start    = off[j + 1] + 1
            
            dim_s = D^(k-j) # Rows in outer product
            dim_e = D^j     # Cols in outer product (contiguous)
            
            push!(updates_k, quote
                for u in 0:$(dim_s - 1)
                    val_s = S[$s_prev_start + u]
                    
                    # Target index in S^k
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
# mul! (Generic arithmetic, Statically Optimized)
# -------------------------------------------------------------------

"""
    mul!(out, x1, x2)
Computes `out = x1 ⊗ x2` generic multiplication (not necessarily group-like).
Used for log-signatures.
"""
@generated function mul!(
    out_tensor::Tensor{T,D,M}, x1_tensor::Tensor{T,D,M}, x2_tensor::Tensor{T,D,M}
) where {T,D,M}
    off = level_starts0(D, M)
    
    level_blocks = Expr[]
    for k in 1:M
        out_len_k = D^k
        out_start = off[k+1] + 1
        
        push!(level_blocks, quote
            # i=0 term: x1^0 * x2^k
            # If x1^0 is 1 or 0, we can optimize, but this is generic.
            if a0 == one(T)
                # Direct copy
                copyto!(out, $out_start, x2, $out_start, $out_len_k)
            elseif a0 == zero(T)
                # Zero out
                @turbo for j in 0:$(out_len_k-1); out[$out_start + j] = zero(T); end
            else
                # Scale
                @turbo for j in 0:$(out_len_k-1); out[$out_start + j] = a0 * x2[$out_start + j]; end
            end
            
            # Middle terms i = 1..k-1
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

            # i=k term: x1^k * x2^0
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
        
        # Level 0
        a0 = x1[$(off[1]+1)]
        b0 = x2[$(off[1]+1)]
        out[$(off[1]+1)] = a0 * b0
        
        @inbounds begin
            $(Expr(:block, level_blocks...))
        end
        return out_tensor
    end
end

# -------------------------------------------------------------------
# Logarithm
# -------------------------------------------------------------------

function log!(out::Tensor{T,D,M}, g::Tensor{T,D,M}) where {T,D,M}
    offsets = out.offsets
    i0 = offsets[1] + 1
    
    # Validate group-like
    # @assert g.coeffs[i0] == one(T) # Assumed true for speed in prod

    X = similar(out)
    copy!(X, g)
    X.coeffs[i0] -= one(T) # X = g - 1

    _zero!(out)
    P = similar(out) # Power accumulator
    Q = similar(out) # Scratch
    copy!(P, X)

    sgn = one(T)
    for k in 1:M
        # out += ((-1)^(k+1) / k) * P
        add_scaled!(out, P, sgn / T(k))
        
        if k < M
            mul!(Q, P, X)
            Q.coeffs[i0] = zero(T)
            P, Q = Q, P # Swap P and Q
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