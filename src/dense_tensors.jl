using StaticArrays
using LoopVectorization: @avx, @turbo

# -------------------------------------------------------------------
# Tensor type
# -------------------------------------------------------------------

struct Tensor{T,D,M} <: AbstractTensor{T}
    coeffs::Vector{T}
    offsets::Vector{Int}
end

dim(::Tensor{T,D,M}) where {T,D,M} = D
level(::Tensor{T,D,M}) where {T,D,M} = M
Base.parent(ts::Tensor) = ts.coeffs
coeffs(ts::Tensor) = ts.coeffs
offsets(ts::Tensor) = ts.offsets
Base.eltype(::Tensor{T,D,M}) where {T,D,M} = T
Base.length(ts::Tensor) = length(ts.coeffs)
@inline Base.getindex(ts::Tensor, i::Int) = @inbounds ts.coeffs[i]
@inline Base.setindex!(ts::Tensor, v, i::Int) = @inbounds (ts.coeffs[i] = v)
Base.show(io::IO, ts::Tensor{T,D,M}) where {T,D,M} = print(io, "Tensor{T=$T, D=$D, M=$M}")

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
    @assert length(coeffs) == offsets[end]
    return Tensor{T,D,M}(coeffs, offsets)
end

function Tensor(coeffs::Vector{T}, d::Int, m::Int) where {T}
    return _make_tensor(coeffs, Val(d), Val(m))
end

@generated function _make_tensor(coeffs::Vector{T}, ::Val{D}, ::Val{M}) where {T,D,M}
    quote; return Tensor{T,D,M}(coeffs); end
end

Base.similar(ts::Tensor{T,D,M}) where {T,D,M} = Tensor{T,D,M}()
Base.copy(ts::Tensor{T,D,M}) where {T,D,M} = Tensor{T,D,M}(copy(ts.coeffs), ts.offsets)
function Base.copy!(dest::Tensor{T,D,M}, src::Tensor{T,D,M}) where {T,D,M}
    copyto!(dest.coeffs, src.coeffs)
    return dest
end

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

function level_starts0(d::Int, m::Int)
    offsets = Vector{Int}(undef, m + 2)
    offsets[1] = 0; len = 1
    @inbounds for k in 1:m+1
        offsets[k+1] = offsets[k] + len
        len *= d
    end
    W = 8; pad = (W - (offsets[2] % W)) % W
    if pad != 0
        @inbounds for k in 2:length(offsets); offsets[k] += pad; end
    end
    return offsets
end

@inline _write_unit!(t::Tensor{T}) where {T} = (t.coeffs[t.offsets[1] + 1] = one(T); t)

# -------------------------------------------------------------------
# Ping-Pong Horner Kernel
# -------------------------------------------------------------------

"""
    update_signature_horner!(A, z, B1, B2)

Updates signature A using increment z.
Uses two scratch buffers B1/B2 to allow strictly forward memory access.
"""
@generated function update_signature_horner!(
    A_tensor::Tensor{T,D,M}, 
    z::SVector{D,T}, 
    B1::AbstractVector{T},
    B2::AbstractVector{T}
) where {T,D,M}
    
    off = level_starts0(D, M)
    updates = Expr[]

    # Main Loop: Update levels k from M down to 2
    for k in M:-1:2
        
        ops = Expr[]
        
        # 1. Initialize B1 (scratch) for this level
        # B1 = z / k
        push!(ops, quote
            inv_k = inv(T($k))
            val_D = $D
            @turbo for d in 1:val_D
                B1[d] = z[d] * inv_k
            end
        end)
        
        current_len = D
        
        # 2. Accumulate intermediate terms (A_i) and expand
        # Loop i from 1 to k-2
        # We ping-pong between B1 and B2
        
        # Track which buffer is 'source' and 'dest' via a compile-time boolean
        # source_is_B1 = true
        
        # Unroll the accumulation loop
        for i in 1:(k-2)
            
            next_scale = inv(T(k - i))
            a_start = off[i+1]
            
            # This iteration: Expand Source -> Dest
            # If i is odd:  Src=B1, Dst=B2
            # If i is even: Src=B2, Dst=B1
            
            src_buf = isodd(i) ? :B1 : :B2
            dst_buf = isodd(i) ? :B2 : :B1
            
            push!(ops, quote
                len = $current_len
                scale = $next_scale
                val_D = $D
                
                # We iterate linearly forward! 
                # src[r] contains the accumulated value for index r
                # We expand it into Dst[ (r-1)*D + 1 ... ]
                
                for r in 1:len
                    # 1. Read accumulator + A_i
                    val = $src_buf[r] + coeffs[$a_start + r]
                    
                    # 2. Write expanded block
                    base_idx = (r-1)*val_D
                    
                    @turbo for c in 1:val_D
                        $dst_buf[base_idx + c] = val * z[c] * scale
                    end
                end
            end)
            
            current_len *= D
        end
        
        # 3. Final Step for Level k
        # Read from the current 'Source' buffer, Add A_{k-1}, Multiply z, Write to A_k
        
        # Determining final source buffer
        # If (k-2) was the last step:
        # If (k-2) was odd, result is in B2.
        # If (k-2) was even (or 0 loops), result is in B1.
        
        last_iter_count = k - 2
        final_src_buf = (last_iter_count > 0 && isodd(last_iter_count)) ? :B2 : :B1
        
        prev_level_idx = k - 1
        a_prev_start = off[prev_level_idx+1]
        a_tgt_start  = off[k+1]
        
        push!(ops, quote
            len = $current_len
            val_D = $D
            
            for r in 1:len
                # val = B_final[r] + A_{k-1}[r]
                val = $final_src_buf[r] + coeffs[$a_prev_start + r]
                
                # Expand directly into A_k
                base_idx = $a_tgt_start + (r-1)*val_D
                
                @turbo for c in 1:val_D
                    coeffs[base_idx + c] += val * z[c]
                end
            end
        end)
        
        push!(updates, Expr(:block, ops...))
    end
    
    # Handle Level 1
    push!(updates, quote
        start_1 = $(off[2])
        val_D = $D
        @turbo for d in 1:val_D
            coeffs[start_1 + d] += z[d]
        end
    end)

    return quote
        coeffs = A_tensor.coeffs
        
        @inbounds begin
            $(Expr(:block, updates...))
        end
        return nothing
    end
end

# -------------------------------------------------------------------
# Fallback / Utility Functions
# -------------------------------------------------------------------

@generated function exp!(out::Tensor{T,D,M}, x::SVector{D,T}) where {T,D,M}
    off = level_starts0(D, M)
    level_loops = Expr[]
    for k in 2:M
        prev_len = D^(k-1)
        prev_s = off[k] + 1; cur_s = off[k+1] + 1
        push!(level_loops, quote
            scale = inv(T($k))
            val_D = $D
            for i in 1:val_D
                val = scale * x[i]
                dest = $cur_s + (i - 1) * $prev_len
                @turbo for j in 0:$(prev_len - 1)
                    coeffs[dest + j] = val * coeffs[$prev_s + j]
                end
            end
        end)
    end
    quote
        coeffs = out.coeffs
        coeffs[$(off[1] + 1)] = one(T)
        s1 = $(off[2] + 1)
        val_D = $D
        @inbounds for i in 1:val_D; coeffs[s1 + i - 1] = x[i]; end
        @inbounds begin; $(Expr(:block, level_loops...)); end
        return nothing
    end
end

@generated function mul!(
    out_tensor::Tensor{T,D,M}, x1_tensor::Tensor{T,D,M}, x2_tensor::Tensor{T,D,M}
) where {T,D,M}
    off = level_starts0(D, M)
    level_blocks = Expr[]
    for k in 1:M
        out_len = D^k; out_s = off[k+1] + 1
        push!(level_blocks, quote
            if a0 == one(T); copyto!(out, $out_s, x2, $out_s, $out_len)
            elseif a0 == zero(T); @turbo for j in 0:$(out_len-1); out[$out_s + j] = zero(T); end
            else; @turbo for j in 0:$(out_len-1); out[$out_s + j] = a0 * x2[$out_s + j]; end; end
            
            $(let inner = Expr[]
                a_len = D
                for i in 1:(k-1)
                    b_len = D^(k-i)
                    a_s = off[i+1] + 1; b_s = off[k-i+1] + 1
                    push!(inner, quote
                        for ai in 0:$(a_len-1)
                            val_a = x1[$a_s + ai]
                            row = $out_s + ai * $b_len
                            @turbo for bi in 0:$(b_len-1); out[row + bi] += val_a * x2[$b_s + bi]; end
                        end
                    end)
                    a_len *= D
                end
                Expr(:block, inner...)
            end)

            if b0 != zero(T)
                if b0 == one(T); @turbo for j in 0:$(out_len-1); out[$out_s + j] += x1[$out_s + j]; end
                else; @turbo for j in 0:$(out_len-1); out[$out_s + j] += b0 * x1[$out_s + j]; end; end
            end
        end)
    end
    quote
        out = out_tensor.coeffs; x1 = x1_tensor.coeffs; x2 = x2_tensor.coeffs
        a0 = x1[$(off[1]+1)]; b0 = x2[$(off[1]+1)]; out[$(off[1]+1)] = a0 * b0
        @inbounds begin; $(Expr(:block, level_blocks...)); end
        return out_tensor
    end
end

function log!(out::Tensor{T,D,M}, g::Tensor{T,D,M}) where {T,D,M}
    i0 = out.offsets[1] + 1
    X = similar(out); copy!(X, g); X.coeffs[i0] -= one(T)
    _zero!(out); P = similar(out); copy!(P, X); Q = similar(out)
    sgn = one(T)
    for k in 1:M
        add_scaled!(out, P, sgn / T(k))
        if k < M
            mul!(Q, P, X); Q.coeffs[i0] = zero(T); P, Q = Q, P
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