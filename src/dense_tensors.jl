using StaticArrays
using LoopVectorization: @avx, @turbo

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

Base.show(io::IO, ts::Tensor{T,D,M}) where {T,D,M} =
    print(io, "Tensor{T=$T, D=$D, M=$M}(length=$(length(ts.coeffs)))")

function Tensor{T,D,M}() where {T,D,M}
    offsets = level_starts0(D, M)
    coeffs  = zeros(T, offsets[end]) 
    return Tensor{T,D,M}(coeffs, offsets)
end

function Tensor{T,D,M}(coeffs::Vector{T}) where {T,D,M}
    offsets = level_starts0(D, M)
    @assert length(coeffs) == offsets[end] "Coefficient length mismatch"
    return Tensor{T,D,M}(coeffs, offsets)
end

Base.similar(ts::Tensor{T,D,M}) where {T,D,M} = Tensor{T,D,M}()
Base.similar(ts::Tensor{T,D,M}, ::Type{S}) where {T,D,M,S} = Tensor{S,D,M}()

Base.copy(ts::Tensor{T,D,M}) where {T,D,M} = 
    Tensor{T,D,M}(copy(ts.coeffs), ts.offsets)

function Base.copy!(dest::Tensor{T,D,M}, src::Tensor{T,D,M}) where {T,D,M}
    copyto!(dest.coeffs, src.coeffs)
    return dest
end

@inline function level_starts0(d::Int, m::Int)
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

@inline function _zero!(ts::Tensor{T}) where {T}
    fill!(ts.coeffs, zero(T))
    return ts
end

@inline function _write_unit!(t::Tensor{T}) where {T}
    t.coeffs[t.offsets[1] + 1] = one(T)
    return t
end

@inline function add_scaled!(dest::Tensor{T,D,M}, src::Tensor{T,D,M}, α::T) where {T,D,M}
    @inbounds @turbo for i in eachindex(dest.coeffs, src.coeffs)
        dest.coeffs[i] = muladd(α, src.coeffs[i], dest.coeffs[i])
    end
    dest
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

function update_signature_horner!(
    A_tensor::Tensor{T,D,M}, 
    z::SVector{D,T}, 
    B1::Vector{T},
    B2::Vector{T}
) where {T,D,M}
    coeffs = A_tensor.coeffs
    off = level_starts0(D, M)
    
    @inbounds begin
        for k in M:-1:2
            inv_k = inv(T(k))
            for d in 1:D
                B1[d] = z[d] * inv_k
            end
            
            current_len = D
            
            for i in 1:(k-2)
                next_scale = inv(T(k - i))
                a_start = off[i+1]
                
                src_buf = isodd(i) ? B1 : B2
                dst_buf = isodd(i) ? B2 : B1
                
                for r in 1:current_len
                    src_val = src_buf[r]
                    coeff_val = coeffs[a_start + r]
                    val = src_val + coeff_val
                    scaled_val = val * next_scale
                    
                    base_idx = (r - 1) * D
                    for d in 1:D
                        dst_buf[base_idx + d] = scaled_val * z[d]
                    end
                end
                
                current_len *= D
            end
            
            last_iter_count = k - 2
            final_src_buf = (last_iter_count > 0 && isodd(last_iter_count)) ? B2 : B1
            
            a_prev_start = off[k-1+1]
            a_tgt_start = off[k+1]
            
            for r in 1:current_len
                src_val = final_src_buf[r]
                coeff_val = coeffs[a_prev_start + r]
                val = src_val + coeff_val
                
                base_idx = (r - 1) * D
                for d in 1:D
                    coeffs[a_tgt_start + base_idx + d] += val * z[d]
                end
            end
        end
        
        start_1 = off[2]
        for d in 1:D
            coeffs[start_1 + d] += z[d]
        end
    end
    return nothing
end