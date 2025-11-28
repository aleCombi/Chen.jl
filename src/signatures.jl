using LinearAlgebra
using StaticArrays

export sig, prepare, logsig, signature_path, signature_path!

# -------------------------------------------------------------------
# 1. Public API (High Level)
# -------------------------------------------------------------------

"""
    sig(path, m)

Compute the truncated signature of the path up to level m.
Returns a flattened array (Python/NumPy friendly).
"""
function sig(path::AbstractMatrix{T}, m::Int) where T
    D = size(path, 2)
    
    # Initialize the structured Tensor
    out = Tensor{T, D, m}()
    
    # Compute signature (using optimized Matrix dispatch)
    signature_path!(out, path)
    
    # Flatten result for Python consumption
    return _flatten_tensor(out)
end

# -------------------------------------------------------------------
# 2. Log-Signature & Preparation
# -------------------------------------------------------------------

struct BasisCache{T}
    d::Int
    m::Int
    lynds::Vector{Algebra.Word}
    L::Matrix{T} 
end

function prepare(d::Int, m::Int)
    lynds, L, _ = Algebra.build_L(d, m)
    return BasisCache(d, m, lynds, L)
end

"""
    logsig(path, basis)

Compute the log-signature projected onto the Lyndon basis.
Optimized to accept Matrix inputs directly.
"""
function logsig(path::AbstractMatrix{T}, basis::BasisCache) where T
    @assert size(path, 2) == basis.d "Dimension mismatch between path and basis"
    
    # Create tensor matching the basis dimensions
    sig_tensor = Tensor{T, basis.d, basis.m}()
    
    # Compute signature (Zero-copy from Matrix)
    signature_path!(sig_tensor, path)
    
    # Compute Logarithm
    log_tensor = ChenSignatures.log(sig_tensor)
    
    # Project to Lyndon Basis
    return Algebra.project_to_lyndon(log_tensor, basis.lynds, basis.L)
end

# -------------------------------------------------------------------
# 3. Allocating Wrapper (Convenience)
# -------------------------------------------------------------------

"""
    signature_path(Type, path, m)

Allocating wrapper that creates a new Tensor and computes the signature.
"""
function signature_path(::Type{Tensor{T}}, path, m::Int) where T
    # Infer D from input type
    D = _get_dim(path)
    out = Tensor{T, D, m}()
    signature_path!(out, path)
    return out
end

_get_dim(path::AbstractMatrix) = size(path, 2)
_get_dim(path::AbstractVector{SVector{D, T}}) where {D, T} = D

# -------------------------------------------------------------------
# 4. Core Logic (Multiple Dispatch)
# -------------------------------------------------------------------

"""
    signature_path!(out, path)

Computes the path signature using Block-Optimized Horner's Method.
Dispatches based on input type to ensure maximum performance.
"""
function signature_path! end

# --- VARIANT A: Optimized for Matrix (NumPy / Standard Arrays) ---
# Reads directly from Matrix, creates SVector on stack. Zero allocations.
function signature_path!(
    out::Tensor{T,D,M},
    path::AbstractMatrix{T}
) where {T,D,M}
    N = size(path, 1)
    @assert N ≥ 2
    
    _reset_tensor!(out)
    B1, B2 = _alloc_scratch(T, D, M)

    @inbounds for i in 1:N-1
        # Create SVector from row difference (Stack allocated, virtually free)
        val = ntuple(j -> path[i+1, j] - path[i, j], Val(D))
        z = SVector{D, T}(val)
        
        ChenSignatures.update_signature_horner!(out, z, B1, B2)
    end
    return out
end

# --- VARIANT B: Optimized for Vector{SVector} (Native Julia) ---
# Uses static array arithmetic directly.
function signature_path!(
    out::Tensor{T,D,M},
    path::AbstractVector{SVector{D,T}}
) where {T,D,M}
    N = length(path)
    @assert N ≥ 2

    _reset_tensor!(out)
    B1, B2 = _alloc_scratch(T, D, M)

    @inbounds for i in 1:N-1
        z = path[i+1] - path[i]
        ChenSignatures.update_signature_horner!(out, z, B1, B2)
    end
    return out
end

# -------------------------------------------------------------------
# 5. Internal Helpers
# -------------------------------------------------------------------

@inline function _reset_tensor!(out::Tensor{T}) where T
    fill!(out.coeffs, zero(T))
    ChenSignatures._write_unit!(out)
end

@inline function _alloc_scratch(::Type{T}, D::Int, M::Int) where T
    # Buffer size needed for Horner's method: D^(M-1)
    max_len = M > 1 ? D^(M-1) : 1
    # We allocate standard Vectors here. 
    # Since this happens once per signature (outside the loop), overhead is negligible.
    return Vector{T}(undef, max_len), Vector{T}(undef, max_len)
end

function _flatten_tensor(t::Tensor{T,D,M}) where {T,D,M}
    # Total size excluding the 0-th level (scalar 1.0)
    total_len = t.offsets[end] - t.offsets[2] 
    out = Vector{T}(undef, total_len)
    
    current_idx = 1
    # Copy level by level
    for k in 1:M
        start_offset = t.offsets[k+1]
        len = D^k
        copyto!(out, current_idx, t.coeffs, start_offset + 1, len)
        current_idx += len
    end
    return out
end