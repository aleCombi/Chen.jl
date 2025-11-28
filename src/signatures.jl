using LinearAlgebra
using StaticArrays

export sig, prepare, logsig, signature_path

"""
    sig(path, m)

Compute the truncated signature of the `path` (Matrix `N×D`) up to level `m`.
Returns a flattened vector suitable for Python/NumPy integration.
"""
function sig(path::AbstractMatrix{T}, m::Int) where T
    D = size(path, 2)
    out = Tensor{T, D, m}()
    signature_path!(out, path)
    return _flatten_tensor(out)
end

struct BasisCache{T}
    d::Int
    m::Int
    lynds::Vector{Word}
    L::Matrix{T} 
end

"""
    prepare(d, m)

Precompute the Lyndon basis and projection matrix for dimension `d` and level `m`.
Returns a `BasisCache` to be passed to `logsig`.
"""
function prepare(d::Int, m::Int)
    lynds, L, _ = build_L(d, m)
    return BasisCache(d, m, lynds, L)
end

"""
    logsig(path, basis)

Compute the log-signature of the `path` projected onto the Lyndon basis.
Requires a precomputed `basis` from `prepare(d, m)`.
"""
function logsig(path::AbstractMatrix{T}, basis::BasisCache) where T
    @assert size(path, 2) == basis.d "Dimension mismatch between path and basis"
    
    sig_tensor = Tensor{T, basis.d, basis.m}()
    signature_path!(sig_tensor, path)
    
    log_tensor = ChenSignatures.log(sig_tensor)
    return project_to_lyndon(log_tensor, basis.lynds, basis.L)
end

function signature_path(::Type{Tensor{T}}, path, m::Int) where T
    D = _get_dim(path)
    out = Tensor{T, D, m}()
    signature_path!(out, path)
    return out
end

_get_dim(path::AbstractMatrix) = size(path, 2)
_get_dim(path::AbstractVector{SVector{D, T}}) where {D, T} = D

function signature_path! end

function signature_path!(out::Tensor{T,D,M}, path::AbstractMatrix{T}) where {T,D,M}
    N = size(path, 1)
    @assert N ≥ 2
    
    _reset_tensor!(out)
    B1, B2 = _alloc_scratch(T, D, M)

    @inbounds for i in 1:N-1
        val = ntuple(j -> path[i+1, j] - path[i, j], Val(D))
        z = SVector{D, T}(val)
        update_signature_horner!(out, z, B1, B2)
    end
    return out
end

function signature_path!(out::Tensor{T,D,M}, path::AbstractVector{SVector{D,T}}) where {T,D,M}
    N = length(path)
    @assert N ≥ 2

    _reset_tensor!(out)
    B1, B2 = _alloc_scratch(T, D, M)

    @inbounds for i in 1:N-1
        z = path[i+1] - path[i]
        update_signature_horner!(out, z, B1, B2)
    end
    return out
end

@inline function _reset_tensor!(out::Tensor{T}) where T
    fill!(out.coeffs, zero(T))
    ChenSignatures._write_unit!(out)
end

@inline function _alloc_scratch(::Type{T}, D::Int, M::Int) where T
    max_len = M > 1 ? D^(M-1) : 1
    return Vector{T}(undef, max_len), Vector{T}(undef, max_len)
end

function _flatten_tensor(t::Tensor{T,D,M}) where {T,D,M}
    total_len = t.offsets[end] - t.offsets[2] 
    out = Vector{T}(undef, total_len)
    
    current_idx = 1
    for k in 1:M
        start_offset = t.offsets[k+1]
        len = D^k
        copyto!(out, current_idx, t.coeffs, start_offset + 1, len)
        current_idx += len
    end
    return out
end