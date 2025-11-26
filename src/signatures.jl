# ---------------- public API ----------------
# Inside module Chen

"""
    signature_path(::Type{AT}, path, m)
"""

# 1. Optimized Entry: User asks for specific Tensor{T,D,M}
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

# 2. Lift Tensor{T} or Tensor{T,M} to Tensor{T,D,M}
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

Computes the signature of `path` storing the result in `out`.
Uses in-place accumulation for maximum efficiency.
"""
function signature_path!(
    out::Tensor{T,D,M},
    path::Vector{SVector{D,T}},
) where {T,D,M}
    @assert length(path) ≥ 2

    # Initialize out to unit (1, 0, 0...)
    _zero!(out)
    _write_unit!(out)

    # Scratch tensor for segment exponential
    seg_tensor = similar(out)

    @inbounds begin
        for i in 1:length(path)-1
            Δ = path[i+1] - path[i]
            
            # 1. Compute exp(Δ) -> seg_tensor
            exp!(seg_tensor, Δ)
            
            # 2. Accumulate: out = out ⊗ seg_tensor
            #    (Done in-place, iterating k=M..1)
            mul_accumulate!(out, seg_tensor)
        end
    end

    return out
end

# Fallback for generic AbstractTensor types
function signature_path!(out::AT, path::Vector{SVector{D,T}}) where {D,T,AT<:AbstractTensor{T}}
    @assert length(path) ≥ 2
    # Generic ping-pong buffer approach
    a = out
    b = similar(out)
    seg = similar(out)
    
    # Init first segment
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