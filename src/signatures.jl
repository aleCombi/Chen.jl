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