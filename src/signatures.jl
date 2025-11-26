# ---------------- public API ----------------

# 1. Optimized Entry
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

# 2. Lift Generic
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

Computes path signature using Horner's Method (pySigLib Alg 2).
Avoids explicit calculation of segment exponentials.
"""
function signature_path!(
    out::Tensor{T,D,M},
    path::Vector{SVector{D,T}},
) where {T,D,M}
    @assert length(path) ≥ 2

    # Initialize out to unit (1, 0, 0...)
    fill!(out.coeffs, zero(T))
    Chen._write_unit!(out)

    # Scratch buffer for Horner's method
    # Must hold intermediate calculation B_k, which grows up to size D^(M-1).
    # Since Tensor{T,D,M} accommodates up to D^M, this is safe.
    B_tensor = similar(out)

    @inbounds begin
        for i in 1:length(path)-1
            Δ = path[i+1] - path[i]
            # Core Horner Update
            update_signature_horner!(out, Δ, B_tensor)
        end
    end

    return out
end

# Fallback for generic AbstractTensor types
function signature_path!(out::AT, path::Vector{SVector{D,T}}) where {D,T,AT<:AbstractTensor{T}}
    @assert length(path) ≥ 2
    a = out
    b = similar(out)
    seg = similar(out)
    Δ1 = path[2] - path[1]
    exp!(a, Δ1)
    for i in 2:length(path)-1
        Δ = path[i+1] - path[i]
        exp!(seg, Δ)
        mul!(b, a, seg)
        a, b = b, a
    end
    if a !== out; copy!(out, a); end
    return out
end