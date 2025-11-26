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
coeffs(ts::Tensor) = ts.coeffs
offsets(ts::Tensor) = ts.offsets

Base.eltype(::Tensor{T,D,M}) where {T,D,M} = T
Base.length(ts::Tensor) = length(ts.coeffs)
Base.getindex(ts::Tensor, i::Int) = ts.coeffs[i]
Base.setindex!(ts::Tensor, v, i::Int) = (ts.coeffs[i] = v)

# Backwards-compatible property access
function Base.getproperty(ts::Tensor{T,D,M}, s::Symbol) where {T,D,M}
    if s === :level
        return M
    elseif s === :dim
        return D
    else
        return getfield(ts, s)
    end
end

Base.show(io::IO, ts::Tensor{T,D,M}) where {T,D,M} =
    print(io, "Tensor{T=$T, D=$D, M=$M}(length=$(length(ts.coeffs)))")

# -------------------------------------------------------------------
# Constructors
# -------------------------------------------------------------------

# 1. Main constructor: Tensor{T,D,M}() - Allocates uninitialized
function Tensor{T,D,M}() where {T,D,M}
    offsets = level_starts0(D, M)
    coeffs  = Vector{T}(undef, offsets[end])
    return Tensor{T,D,M}(coeffs, offsets)
end

# 2. Constructor with existing data (internal use primarily)
function Tensor{T,D,M}(coeffs::Vector{T}) where {T,D,M}
    offsets = level_starts0(D, M)
    @assert length(coeffs) == offsets[end] "Coefficient length mismatch"
    return Tensor{T,D,M}(coeffs, offsets)
end

# 3. Dynamic factory (returns a specific type instance based on runtime args)
#    Warning: This is type-unstable if D/M are not constants.
function Tensor(coeffs::Vector{T}, d::Int, m::Int) where {T}
    # Dispatch to the static type constructor using value-to-type bridge
    # This might trigger dynamic dispatch, which is expected for runtime values.
    return _make_tensor(coeffs, Val(d), Val(m))
end

@generated function _make_tensor(coeffs::Vector{T}, ::Val{D}, ::Val{M}) where {T,D,M}
    quote
        return Tensor{T,D,M}(coeffs)
    end
end

# 4. Similar
Base.similar(ts::Tensor{T,D,M}) where {T,D,M} = Tensor{T,D,M}()
Base.similar(ts::Tensor{T,D,M}, ::Type{S}) where {T,D,M,S} = Tensor{S,D,M}()

Base.copy(ts::Tensor{T,D,M}) where {T,D,M} = 
    Tensor{T,D,M}(copy(ts.coeffs), ts.offsets)

function Base.copy!(dest::Tensor{T,D,M}, src::Tensor{T,D,M}) where {T,D,M}
    copyto!(dest.coeffs, src.coeffs)
    return dest
end

# -------------------------------------------------------------------
# Equality
# -------------------------------------------------------------------

function Base.isapprox(a::Tensor{Ta,Da,Ma}, b::Tensor{Tb,Db,Mb};
                       atol::Real=1e-8, rtol::Real=1e-8) where {Ta,Tb,Da,Db,Ma,Mb}
    Da == Db && Ma == Mb || return false
    sA, sB = a.offsets, b.offsets
    d = Da
    len = 1
    @inbounds begin
        a0 = a.coeffs[sA[1] + 1]; b0 = b.coeffs[sB[1] + 1]
        if !(abs(a0 - b0) <= atol + rtol*max(abs(a0),abs(b0))); return false; end
        for k in 1:Ma
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
# Small helpers
# -------------------------------------------------------------------

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

function lift1!(ts::Tensor{T,D,M}, x::AbstractVector{T}) where {T,D,M}
    @assert length(x) == D
    fill!(ts.coeffs, zero(T))
    @inbounds copyto!(ts.coeffs, 1, x, 1, D)
    return ts
end

# -------------------------------------------------------------------
# exp! (Specialized for SVector{D} and Tensor{T,D,M})
# -------------------------------------------------------------------

"""
    exp!(out::Tensor{T,D,M}, x::SVector{D,T})

Fully unrolled tensor exponential.
"""
@generated function exp!(out::Tensor{T,D,M}, x::SVector{D,T}) where {T,D,M}
    level_loops = Expr[]
    
    for k in 2:M
        prev_len_val = D^(k-1)
        push!(level_loops, quote
            prev_start = offsets[$k] + 1
            cur_start  = offsets[$(k+1)] + 1
            scale      = inv(T($k))
            
            for i in 1:D
                val = scale * x[i]
                dest_ptr = cur_start + (i - 1) * $prev_len_val
                @avx for j in 0:$(prev_len_val - 1)
                    coeffs[dest_ptr + j] = val * coeffs[prev_start + j]
                end
            end
        end)
    end

    quote
        coeffs = out.coeffs
        offsets = out.offsets
        
        # Level 0
        coeffs[offsets[1] + 1] = one(T)
        
        # Level 1
        start1 = offsets[2] + 1
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
# mul_grouplike! (Specialized)
# -------------------------------------------------------------------

@generated function mul_grouplike!(
    out_tensor::Tensor{T,D,M}, x1_tensor::Tensor{T,D,M}, x2_tensor::Tensor{T,D,M}
) where {T,D,M}
    level_blocks = Expr[]
    for k in 1:M
        out_len_k = D^k
        
        push!(level_blocks, quote
            # ---- level $k ----
            out_start = offsets[$k + 1] + 1

            # Base: out = x1 + x2
            @avx for j in 0:$(out_len_k - 1)
                out[out_start + j] = x1[out_start + j] + x2[out_start + j]
            end

            # Middle terms: i = 1..k-1
            $(let inner_loops = Expr[]
                a_len = D
                for i in 1:(k-1)
                    b_len = D^(k-i)
                    push!(inner_loops, quote
                        a_block_start = offsets[$(i + 1)]
                        b_block_start = offsets[$(k - i + 1)]
                        
                        for ai in 1:$a_len
                            val_a = x1[a_block_start + ai]
                            row0 = out_start + (ai - 1) * $b_len - 1
                            @avx for bi in 1:$b_len
                                out[row0 + bi] = muladd(
                                    val_a,
                                    x2[b_block_start + bi],
                                    out[row0 + bi],
                                )
                            end
                        end
                    end)
                    a_len *= D
                end
                Expr(:block, inner_loops...)
            end)
        end)
    end

    return quote
        out = out_tensor.coeffs
        x1  = x1_tensor.coeffs
        x2  = x2_tensor.coeffs
        offsets = out_tensor.offsets

        out[offsets[1] + 1] = one(T) # Level 0

        @inbounds begin
            $(Expr(:block, level_blocks...))
        end
        out_tensor
    end
end

# -------------------------------------------------------------------
# mul! (Generic arithmetic, statically sized)
# -------------------------------------------------------------------

@inline function mul!(
    out_tensor::Tensor{T,D,M}, x1_tensor::Tensor{T,D,M}, x2_tensor::Tensor{T,D,M}
) where {T,D,M}
    out, x1, x2 = out_tensor.coeffs, x1_tensor.coeffs, x2_tensor.coeffs
    offsets = out_tensor.offsets
    
    a0 = x1[offsets[1] + 1]
    b0 = x2[offsets[1] + 1]
    out[offsets[1] + 1] = a0 * b0

    out_len = D
    @inbounds for k in 1:M
        out_start = offsets[k + 1] + 1
        b_start = offsets[k + 1]
        
        # i=0 term
        if a0 == one(T)
            copyto!(out, out_start, x2, b_start + 1, out_len)
        elseif a0 == zero(T)
            @avx for j in 1:out_len; out[out_start + j - 1] = zero(T); end
        else
            @avx for j in 1:out_len; out[out_start + j - 1] = a0 * x2[b_start + j]; end
        end

        # i=1..k-1
        a_len = D
        for i in 1:(k-1)
            a_start = offsets[i + 1]
            b_start = offsets[k - i + 1]
            b_len   = out_len ÷ a_len

            @avx for ai in 1:a_len, bi in 1:b_len
                row0 = out_start + (ai - 1) * b_len - 1
                out[row0 + bi] = muladd(x1[a_start + ai], x2[b_start + bi], out[row0 + bi])
            end
            a_len *= D
        end

        # i=k term
        if b0 != zero(T)
            a_start = offsets[k + 1]
            if b0 == one(T)
                @avx for j in 1:out_len; out[out_start + j - 1] += x1[a_start + j]; end
            else
                @avx for j in 1:out_len; out[out_start + j - 1] = muladd(b0, x1[a_start + j], out[out_start + j - 1]); end
            end
        end

        out_len *= D
    end
    return out
end

# -------------------------------------------------------------------
# Helpers (Offsets & Log)
# -------------------------------------------------------------------

function level_starts0(d::Int, m::Int)
    offsets = Vector{Int}(undef, m + 2)
    offsets[1] = 0
    len = 1
    @inbounds for k in 1:m+1
        offsets[k+1] = offsets[k] + len
        len *= d
    end
    W = 8
    pad = (W - (offsets[2] % W)) % W
    if pad != 0
        @inbounds for k in 2:length(offsets)
            offsets[k] += pad
        end
    end
    return offsets
end

function log!(out::Tensor{T,D,M}, g::Tensor{T,D,M}) where {T,D,M}
    offsets = out.offsets
    i0 = offsets[1] + 1
    @assert g.coeffs[i0] == one(T)

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