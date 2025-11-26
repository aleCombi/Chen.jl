using LinearAlgebra
export sig, prepare, logsig

# --- 1. Signature (sig) ---
"""
    sig(path, m)
Matches iisignature.sig(X, m).
Returns a flattened Vector of the signature of path X up to level m.
"""
function sig(path::AbstractMatrix{T}, m::Int) where T
    # Convert Matrix (N x d) to Vector of SVectors
    N, d = size(path)
    sv_path = [SVector{d, T}(path[i,:]) for i in 1:N]
    
    # Compute using the Dense backend, preserving type T
    tensor = signature_path(Tensor{T}, sv_path, m)
    
    # Flatten the tensor
    return _flatten_tensor(tensor)
end

# --- 2. Preparation (prepare) ---
struct BasisCache{T}
    d::Int
    m::Int
    lynds::Vector{Algebra.Word}
    L::Matrix{T} 
end

"""
    prepare(d, m)
Returns a BasisCache object containing the Lyndon basis projection matrix.
"""
function prepare(d::Int, m::Int)
    lynds, L, _ = Algebra.build_L(d, m)
    return BasisCache(d, m, lynds, L)
end

# --- 3. Log Signature (logsig) ---
"""
    logsig(path, basis)
Computes the log-signature projected onto the Lyndon basis.
"""
function logsig(path::AbstractMatrix{T}, basis::BasisCache) where T
    N, d = size(path)
    @assert d == basis.d "Dimension mismatch between path and basis"
    
    sv_path = [SVector{d, T}(path[i,:]) for i in 1:N]
    
    # 1. Compute full signature (preserving type T)
    sig_tensor = signature_path(Tensor{T}, sv_path, basis.m)
    
    # 2. Compute tensor logarithm
    log_tensor = Chen.log(sig_tensor)
    
    # 3. Project to Lyndon basis using the precomputed matrix L
    return Algebra.project_to_lyndon(log_tensor, basis.lynds, basis.L)
end

# --- Helper: Flatten Tensor to Array ---
function _flatten_tensor(t::Tensor{T,D,M}) where {T,D,M}
    total_len = t.offsets[end] - t.offsets[2] # skip level 0
    out = Vector{T}(undef, total_len)
    
    current_idx = 1
    
    for k in 1:M
        start_offset = t.offsets[k+1]
        len = D^k
        # Copy raw coefficients for this level
        copyto!(out, current_idx, t.coeffs, start_offset + 1, len)
        current_idx += len
    end
    return out
end