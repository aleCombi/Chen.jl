# ---------------- public API ----------------

function signature_path(
    ::Type{Tensor{T,D,M}},
    path::Vector{SVector{D,T}},
    m::Int,
) where {T,D,M}
    @assert m == M "Requested level m=$m does not match Type level M=$M"
    out = Tensor{T,D,M}()
    signature_path!(out, path)
    return out
end

function signature_path(::Type{Tensor{T}}, path::Vector{SVector{D,T}}, m::Int) where {T,D}
    return _dispatch_sig(Tensor{T}, Val(D), Val(m), path)
end
function signature_path(::Type{Tensor{T,M}}, path::Vector{SVector{D,T}}, m::Int) where {T,D,M}
    return _dispatch_sig(Tensor{T}, Val(D), Val(M), path)
end

@generated function _dispatch_sig(::Type{Tensor{T}}, ::Val{D}, ::Val{M}, path) where {T,D,M}
    quote
        out = Tensor{T,D,M}()
        signature_path!(out, path)
        return out
    end
end

"""
    signature_path!(out, path)
Computes path signature using Block-Optimized Horner's Method.
"""
function signature_path!(
    out::Tensor{T,D,M},
    path::Vector{SVector{D,T}},
) where {T,D,M}
    @assert length(path) ≥ 2

    fill!(out.coeffs, zero(T))
    ChenSignatures._write_unit!(out)

    # Scratch buffers for Ping-Pong
    # Size: D^(M-1) floats
    max_scratch_len = M > 1 ? D^(M-1) : 1
    
    B1 = Vector{T}(undef, max_scratch_len)
    B2 = Vector{T}(undef, max_scratch_len)

    @inbounds begin
        for i in 1:length(path)-1
            Δ = path[i+1] - path[i]
            update_signature_horner!(out, Δ, B1, B2)
        end
    end

    return out
end

function signature_path!(out::AT, path::Vector{SVector{D,T}}) where {D,T,AT<:AbstractTensor{T}}
    @assert length(path) ≥ 2
    a = out; b = similar(out); seg = similar(out)
    Δ1 = path[2] - path[1]; exp!(a, Δ1)
    for i in 2:length(path)-1
        Δ = path[i+1] - path[i]; exp!(seg, Δ); mul!(b, a, seg); a, b = b, a
    end
    if a !== out; copy!(out, a); end
    return out
end

# src/signatures.jl

# src/signatures.jl

# src/signatures.jl or src/api.jl

# src/signatures.jl or wherever you put signature_from_matrix

# src/signatures.jl

"""
    signature_from_matrix!(out, seg, tmp, Δ_vec, path_matrix, m)
    
Enzyme-compatible: All buffers pre-allocated outside differentiated region.
"""
@inline function signature_from_matrix!(
    out::Tensor{T,D,M},
    seg::Tensor{T,D,M},
    tmp::Tensor{T,D,M},
    Δ_vec::Vector{T},
    path_matrix::Matrix{T},
    ::Val{M}
) where {T,D,M}
    
    N = size(path_matrix, 1)
    
    # Initialize out
    @inbounds for i in eachindex(out.coeffs)
        out.coeffs[i] = zero(T)
    end
    out.coeffs[out.offsets[1] + 1] = one(T)
    
    # First segment
    @inbounds for j in 1:D
        Δ_vec[j] = path_matrix[2, j] - path_matrix[1, j]
    end
    non_generated_exp!(seg, Δ_vec)
    
    # Manual copy
    @inbounds for i in eachindex(out.coeffs, seg.coeffs)
        out.coeffs[i] = seg.coeffs[i]
    end
    
    # Remaining segments
    @inbounds for i in 2:N-1
        for j in 1:D
            Δ_vec[j] = path_matrix[i+1, j] - path_matrix[i, j]
        end
        
        non_generated_exp!(seg, Δ_vec)
        non_generated_mul!(tmp, out, seg)
        
        # Manual copy
        for k in eachindex(out.coeffs, tmp.coeffs)
            out.coeffs[k] = tmp.coeffs[k]
        end
    end
    
    return out
end

# Convenience wrapper that allocates (for non-AD use)
@inline function signature_from_matrix(path_matrix::Matrix{T}, m::Int) where {T}
    N, D = size(path_matrix)
    return _sig_from_matrix_alloc(path_matrix, Val(D), Val(m))
end

@inline function _sig_from_matrix_alloc(
    path_matrix::Matrix{T}, ::Val{D}, ::Val{M}
) where {T,D,M}
    out = Tensor{T,D,M}()
    seg = Tensor{T,D,M}()
    tmp = Tensor{T,D,M}()
    Δ_vec = Vector{T}(undef, D)
    
    signature_from_matrix!(out, seg, tmp, Δ_vec, path_matrix, Val(M))
    return out
end