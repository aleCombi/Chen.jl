using LinearAlgebra

export sig, prepare, logsig

# --- 1. Signature (sig) ---

"""
    sig(path, m)

Matches iisignature.sig(X, m).
Returns a flattened Vector{Float64} of the signature of path X up to level m.
"""
function sig(path::AbstractMatrix, m::Int)
    # Convert Matrix (N x d) to Vector of SVectors for Chen's optimization
    N, d = size(path)
    # Note: Chen expects SVectors for maximum speed
    sv_path = [SVector{d, Float64}(path[i,:]) for i in 1:N]
    
    # Compute using the Dense backend
    tensor = signature_path(Tensor{Float64}, sv_path, m)
    
    # Flatten the tensor (Chen stores it structurally, iisig returns a flat array)
    # We skip the 0-th level (constant 1.0) to match iisignature
    return _flatten_tensor(tensor)
end

# --- 2. Preparation (prepare) ---

struct BasisCache{T}
    d::Int
    m::Int
    # FIXED: Use Algebra.Word
    lynds::Vector{Algebra.Word}
    L::Matrix{T} # Projection matrix to Lyndon basis
end

"""
    prepare(d, m)

Matches iisignature.prepare(d, m).
Returns a BasisCache object containing the Lyndon basis projection matrix.
"""
function prepare(d::Int, m::Int)
    # FIXED: Call Algebra.build_L
    lynds, L, _ = Algebra.build_L(d, m)
    return BasisCache(d, m, lynds, L)
end

# --- 3. Log Signature (logsig) ---

"""
    logsig(path, basis)

Matches iisignature.logsig(X, s).
Computes the log-signature projected onto the Lyndon basis.
"""
function logsig(path::AbstractMatrix, basis::BasisCache)
    N, d = size(path)
    @assert d == basis.d "Dimension mismatch between path and basis"
    
    sv_path = [SVector{d, Float64}(path[i,:]) for i in 1:N]
    
    # 1. Compute full signature
    sig_tensor = signature_path(Tensor{Float64}, sv_path, basis.m)
    
    # 2. Compute tensor logarithm (Chen.log is generic)
    log_tensor = Chen.log(sig_tensor)
    
    # 3. Project to Lyndon basis using the precomputed matrix L
    # FIXED: Call Algebra.project_to_lyndon
    return Algebra.project_to_lyndon(log_tensor, basis.lynds, basis.L)
end

# --- Helper: Flatten Tensor to Array ---
function _flatten_tensor(t::Tensor{T}) where T
    # iisignature returns a single flat array of all levels concatenated
    # Chen stores them with offsets.
    total_len = t.offsets[end] - t.offsets[2] # skip level 0
    out = Vector{T}(undef, total_len)
    
    current_idx = 1
    d = t.dim
    
    for k in 1:t.level
        start_offset = t.offsets[k+1]
        len = d^k
        # Copy raw coefficients for this level
        copyto!(out, current_idx, t.coeffs, start_offset + 1, len)
        current_idx += len
    end
    return out
end