using LinearAlgebra
export sig, prepare, logsig, sig_enzyme

# --- 1. Signature (sig) ---
function sig(path::AbstractMatrix{T}, m::Int) where T
    N, d = size(path)
    sv_path = [SVector{d, T}(path[i,:]) for i in 1:N]
    
    # Use Tensor{T} to match input type
    tensor = signature_path(Tensor{T}, sv_path, m)
    
    return _flatten_tensor(tensor)
end

# Return flattened tensor (no sum)
@inline function sig_enzyme(path_matrix::Matrix{Float64}, m::Int)
    D = size(path_matrix, 2)
    M = m
    N = size(path_matrix, 1)
    Δ = Vector{Float64}(undef, D)
    
    a = ChenSignatures.Tensor{Float64,D,M}()
    b = ChenSignatures.Tensor{Float64,D,M}()
    seg = ChenSignatures.Tensor{Float64,D,M}()
    
    @inbounds for j in 1:D
        Δ[j] = path_matrix[2, j] - path_matrix[1, j]
    end
    ChenSignatures.non_generated_exp_vec!(a, Δ)
    
    @inbounds for i in 2:N-1
        for j in 1:D
            Δ[j] = path_matrix[i+1, j] - path_matrix[i, j]
        end
        
        ChenSignatures.non_generated_exp_vec!(seg, Δ)
        ChenSignatures.non_generated_mul!(b, a, seg)
        
        a, b = b, a
    end
    
    return ChenSignatures._flatten_tensor(a)  # Return vector, not scalar
end

# --- 2. Preparation (prepare) ---
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

# --- 3. Log Signature (logsig) ---
function logsig(path::AbstractMatrix{T}, basis::BasisCache) where T
    N, d = size(path)
    @assert d == basis.d "Dimension mismatch between path and basis"
    
    sv_path = [SVector{d, T}(path[i,:]) for i in 1:N]
    
    sig_tensor = signature_path(Tensor{T}, sv_path, basis.m)
    log_tensor = ChenSignatures.log(sig_tensor)
    
    return Algebra.project_to_lyndon(log_tensor, basis.lynds, basis.L)
end

# --- Helper: Flatten Tensor to Array ---
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