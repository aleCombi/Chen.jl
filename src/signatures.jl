# ---------------- public API ----------------
# In src/signatures.jl, inside module Chen

"""
    signature_path(::Type{AT}, path, m)

Compute the truncated signature of a piecewise-linear path `path`.
"""

# 1. Fully specified case: User asks for Tensor{T,D,M}
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

# 2. Generic Tensor{T} or Tensor{T,M} passed: Lift D and m to type parameters
function signature_path(
    ::Type{Tensor{T}},
    path::Vector{SVector{D,T}},
    m::Int,
) where {T,D}
    return _dispatch_sig(Tensor{T}, Val(D), Val(m), path)
end

function signature_path(
    ::Type{Tensor{T,M}},
    path::Vector{SVector{D,T}},
    m::Int,
) where {T,D,M}
    @assert m == M
    return _dispatch_sig(Tensor{T}, Val(D), Val(M), path)
end

# Dispatch barrier to create the concrete type
@generated function _dispatch_sig(::Type{Tensor{T}}, ::Val{D}, ::Val{M}, path) where {T,D,M}
    quote
        out = Tensor{T,D,M}()
        signature_path!(out, path)
        return out
    end
end

"""
    signature_path!(out::Tensor{T,D,M}, path)

Highly optimized signature calculation.
"""
function signature_path!(
    out::Tensor{T,D,M},
    path::Vector{SVector{D,T}},
) where {T,D,M}
    @assert length(path) ≥ 2

    a = out
    b = similar(out)
    segment_tensor = similar(out)

    @inbounds begin
        # First segment
        Δ = path[2] - path[1]
        exp!(a, Δ) 

        # Remaining segments
        for i in 2:length(path)-1
            Δ = path[i+1] - path[i]
            exp!(segment_tensor, Δ)
            mul_grouplike!(b, a, segment_tensor) # No Val(D) needed, D is in type
            a, b = b, a
        end
    end

    if a !== out
        copy!(out, a)
    end

    return out
end

# Generic fallback
function signature_path!(
    out::AT,
    path::Vector{SVector{D,T}},
) where {D,T,AT<:AbstractTensor{T}}
    @assert length(path) ≥ 2
    a = out
    b = similar(out)
    segment_tensor = similar(out)

    @inbounds begin
        Δ = path[2] - path[1]
        exp!(a, Δ)
        for i in 2:length(path)-1
            Δ = path[i+1] - path[i]
            exp!(segment_tensor, Δ)
            mul!(b, a, segment_tensor)
            a, b = b, a
        end
    end
    if a !== out; copy!(out, a); end
    return out
end