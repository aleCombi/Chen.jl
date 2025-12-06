using StaticArrays

"""
    Tensor{T,D,M} <: AbstractTensor{T}

Dense tensor algebra element up to level `M` in dimension `D`.

This is the core data structure for representing truncated tensor series in the path
signature computation. It stores coefficients for all tensor levels from 0 to `M` in a
single flat array with efficient memory layout.

# Type Parameters
- `T`: Element type (e.g., `Float64`, `Float32`)
- `D`: Dimension (number of coordinate axes)
- `M`: Maximum truncation level

# Fields
- `coeffs::Vector{T}`: Flattened coefficient array containing all levels 0 through `M`.
  Length is `1 + D + D² + ... + Dᴹ⁺¹` (includes padding for alignment).
- `offsets::Vector{Int}`: Starting indices for each level in `coeffs`. Length `M+2`.

# Construction
```julia
# Create zero tensor
t = Tensor{Float64, 3, 4}()

# Create from coefficient vector
coeffs = randn(len)  # Must match expected length
t = Tensor{Float64, 3, 4}(coeffs)
```

# Notes
- Most users should use [`sig`](@ref) instead of working with `Tensor` directly.
- For advanced applications requiring direct tensor manipulation, see [`signature_path!`](@ref).
- The type parameters `{T,D,M}` are compile-time constants, enabling aggressive optimization.

See also: [`sig`](@ref), [`signature_path`](@ref), [`SignatureWorkspace`](@ref)
"""
struct Tensor{T,D,M,V<:AbstractVector{T}} <: AbstractTensor{T}
    coeffs::V
    offsets::Vector{Int}
end

dim(::Tensor{T,D,M,V}) where {T,D,M,V} = D
level(::Tensor{T,D,M,V}) where {T,D,M,V} = M

Base.parent(ts::Tensor) = ts.coeffs
coeffs(ts::Tensor) = ts.coeffs
offsets(ts::Tensor) = ts.offsets

Base.eltype(::Tensor{T,D,M,V}) where {T,D,M,V} = T
Base.length(ts::Tensor) = length(ts.coeffs)
@inline Base.getindex(ts::Tensor, i::Int) = @inbounds ts.coeffs[i]
@inline Base.setindex!(ts::Tensor, v, i::Int) = @inbounds (ts.coeffs[i] = v)

Base.show(io::IO, ts::Tensor{T,D,M,V}) where {T,D,M,V} =
    print(io, "Tensor{T=$T, D=$D, M=$M, V=$V}(length=$(length(ts.coeffs)))")

function Tensor{T,D,M}() where {T,D,M}
    offsets = level_starts0(D, M)
    coeffs  = zeros(T, offsets[end])
    return Tensor{T,D,M,typeof(coeffs)}(coeffs, offsets)
end

function Tensor{T,D,M}(coeffs::V) where {T,D,M,V<:AbstractVector{T}}
    offsets = level_starts0(D, M)
    @assert length(coeffs) == offsets[end] "Coefficient length mismatch"
    return Tensor{T,D,M,V}(coeffs, offsets)
end

Base.similar(ts::Tensor{T,D,M,V}) where {T,D,M,V} = Tensor{T,D,M}(similar(ts.coeffs))
Base.similar(ts::Tensor{T,D,M,V}, ::Type{S}) where {T,D,M,V,S} = Tensor{S,D,M}()

Base.copy(ts::Tensor{T,D,M,V}) where {T,D,M,V} =
    Tensor{T,D,M}(copy(ts.coeffs))

function Base.copy!(dest::Tensor{T,D,M,V1}, src::Tensor{T,D,M,V2}) where {T,D,M,V1,V2}
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

@inline function _flatten_tensor_core!(
    results::AbstractMatrix{T},
    tensor_coeffs_all::AbstractMatrix{T},
    offsets::AbstractVector{Int},
    ::Val{D},
    ::Val{M},
    col_idx::Int
) where {D,M,T}
    idx_out = 1
    @inbounds for k in 1:M
        start_offset = offsets[k + 1]
        len = D^k
        for j in 1:len
            results[idx_out, col_idx] = tensor_coeffs_all[start_offset + j, col_idx]
            idx_out += 1
        end
    end
    return nothing
end

@inline function _horner_update_core!(
    tensor_coeffs_all::AbstractMatrix{T},
    ws_B1::AbstractMatrix{T},
    ws_B2::AbstractMatrix{T},
    offsets::AbstractVector{Int},
    inv_level::AbstractVector{T},
    z::NTuple{D,T},
    ::Val{D},
    ::Val{M},
    coeff_col_idx::Int,
    ws_col_idx::Int
) where {D,M,T}
    @inbounds begin
        for k in M:-1:2
            inv_k = inv_level[k]
            for d in 1:D
                ws_B1[d, ws_col_idx] = z[d] * inv_k
            end

            current_len = D

            for iter in 1:(k-2)
                inv_next = inv_level[k - iter]
                a_start = offsets[iter + 1]
                src_is_B1 = isodd(iter)

                for r in 1:current_len
                    src_val = src_is_B1 ? ws_B1[r, ws_col_idx] : ws_B2[r, ws_col_idx]
                    coeff_val = tensor_coeffs_all[a_start + r, coeff_col_idx]
                    val = src_val + coeff_val
                    scaled = val * inv_next

                    base_idx = (r - 1) * D
                    for d in 1:D
                        idx = base_idx + d
                        if src_is_B1
                            ws_B2[idx, ws_col_idx] = scaled * z[d]
                        else
                            ws_B1[idx, ws_col_idx] = scaled * z[d]
                        end
                    end
                end

                current_len *= D
            end

            last_iter_count = k - 2
            use_B2 = (last_iter_count > 0 && isodd(last_iter_count))
            a_prev_start = offsets[k]
            a_tgt_start = offsets[k + 1]

            for r in 1:current_len
                src_val = use_B2 ? ws_B2[r, ws_col_idx] : ws_B1[r, ws_col_idx]
                coeff_val = tensor_coeffs_all[a_prev_start + r, coeff_col_idx]
                val = src_val + coeff_val

                base_idx = (r - 1) * D
                for d in 1:D
                    idx = a_tgt_start + base_idx + d
                    tensor_coeffs_all[idx, coeff_col_idx] += val * z[d]
                end
            end
        end

        start_1 = offsets[2]
        for d in 1:D
            tensor_coeffs_all[start_1 + d, coeff_col_idx] += z[d]
        end
    end
    return nothing
end

@inline function add_scaled!(dest::Tensor{T,D,M,V1}, src::Tensor{T,D,M,V2}, α::T) where {T,D,M,V1,V2}
    off = dest.offsets

    # Level 0
    idx0 = off[1] + 1
    @inbounds dest.coeffs[idx0] = muladd(α, src.coeffs[idx0], dest.coeffs[idx0])

    # Levels 1 to M
    @inbounds for k in 1:M
        len = D^k
        start = off[k+1] + 1
        @simd for i in 0:(len-1)
            dest.coeffs[start + i] = muladd(α, src.coeffs[start + i], dest.coeffs[start + i])
        end
    end

    return dest
end

@inline function mul!(
    out_tensor::Tensor{T,D,M,V1},
    x1_tensor::Tensor{T,D,M,V2},
    x2_tensor::Tensor{T,D,M,V3}
) where {T,D,M,V1,V2,V3}
    out = out_tensor.coeffs
    x1  = x1_tensor.coeffs
    x2  = x2_tensor.coeffs
    off = out_tensor.offsets
    
    # Level 0
    idx0 = off[1] + 1
    a0 = x1[idx0]
    b0 = x2[idx0]
    @inbounds out[idx0] = a0 * b0

    @inbounds for k in 1:M
        out_len = D^k
        out_s   = off[k+1] + 1
        
        # 1. Term: x1[0] * x2[k]
        start_x2 = off[k+1] + 1
        if a0 == one(T)
            copyto!(out, out_s, x2, start_x2, out_len)
        elseif a0 == zero(T)
            fill!(view(out, out_s:(out_s+out_len-1)), zero(T))
        else
            @simd for j in 0:(out_len-1)
                out[out_s + j] = a0 * x2[start_x2 + j]
            end
        end

        # 2. Convolution Terms: x1[i] * x2[k-i]
        a_len = D
        for i in 1:(k-1)
            b_len = D^(k-i)
            a_s = off[i+1] + 1
            b_s = off[k-i+1] + 1
            
            for ai in 0:(a_len-1)
                val_a = x1[a_s + ai]
                row_start = out_s + ai * b_len
                # Inner loop over b_len
                @simd for bi in 0:(b_len-1)
                    out[row_start + bi] = muladd(val_a, x2[b_s + bi], out[row_start + bi])
                end
            end
            a_len *= D
        end

        # 3. Term: x1[k] * x2[0]
        if b0 != zero(T)
            start_x1 = off[k+1] + 1
            if b0 == one(T)
                @simd for j in 0:(out_len-1)
                    out[out_s + j] += x1[start_x1 + j]
                end
            else
                @simd for j in 0:(out_len-1)
                    out[out_s + j] = muladd(b0, x1[start_x1 + j], out[out_s + j])
                end
            end
        end
    end
    return out_tensor
end

function log!(out::Tensor{T,D,M,V1}, g::Tensor{T,D,M,V2}) where {T,D,M,V1,V2}
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

function log(g::Tensor{T,D,M,V}) where {T,D,M,V}
    out = similar(g)
    return log!(out, g)
end

@inline function exp!(out::Tensor{T,D,M,V}, x::SVector{D,T}) where {T,D,M,V}
    coeffs = out.coeffs
    off = out.offsets

    @inbounds coeffs[off[1] + 1] = one(T)

    s1 = off[2] + 1
    @inbounds for i in 1:D
        coeffs[s1 + i - 1] = x[i]
    end

    prev_len = D
    @inbounds for k in 2:M
        scale = inv(T(k))
        prev_s = off[k] + 1
        cur_s  = off[k+1] + 1
        
        for i in 1:D
            val = scale * x[i]
            dest_offset = cur_s + (i - 1) * prev_len
            
            @simd for j in 0:(prev_len - 1)
                coeffs[dest_offset + j] = val * coeffs[prev_s + j]
            end
        end
        prev_len *= D
    end
    return nothing
end

function update_signature_horner!(
    A_tensor::Tensor{T,D,M,V1},
    z::SVector{D,T},
    B1::V2,
    B2::V3
) where {T,D,M,V1<:AbstractVector{T},V2<:AbstractVector{T},V3<:AbstractVector{T}}
    inv_level = [inv(T(k)) for k in 1:M]
    coeffs_mat = reshape(A_tensor.coeffs, :, 1)
    ws1_mat = reshape(B1, :, 1)
    ws2_mat = reshape(B2, :, 1)
    _horner_update_core!(coeffs_mat, ws1_mat, ws2_mat, A_tensor.offsets, inv_level, Tuple(z), Val(D), Val(M), 1, 1)
    return nothing
end
