# ---------------- public API ----------------
# In src/signatures.jl, inside module Chen

"""
    signature_path(::Type{AT}, path, m)

Compute the truncated signature of a piecewise-linear path `path`
up to level `m`, using tensor backend `AT<:AbstractTensor{T}`.

For Chen.Tensor, a specialised method uses the fixed-level Tensor{T,M}
backend for better performance when the level is known at compile time.
"""
function signature_path(
    ::Type{AT},
    path::Vector{SVector{D,T}},
    m::Int,
) where {D,T,AT<:AbstractTensor{T}}

    @assert length(path) ≥ 2 "path must have at least 2 points"

    # Generic backend: assume AT(dim, level) constructor
    out = AT(D, m)
    signature_path!(out, path)
    return out
end

# Specialised constructor for Chen.Tensor{T,M}:
# level is a type parameter, ignore/validate the passed `m`.
function signature_path(
    ::Type{Tensor{T,M}},
    path::Vector{SVector{D,T}},
    m::Int,
) where {D,T,M}

    @assert length(path) ≥ 2 "path must have at least 2 points"
    @assert m == M "requested level m=$m must match Tensor level M=$M"

    out = Tensor{T,M}(D)
    signature_path!(out, path)
    return out
end


"""
    signature_path!(out, path)

In-place version: writes the signature of `path` into `out`.
The behaviour depends on the tensor backend:

  * For Chen.Tensor{T,M}: uses `mul_grouplike!` and exploits that each
    segment exponential is group-like.
  * For other AbstractTensor backends: falls back to `mul!`.
"""

# Specialised, fast path for Chen.Tensor{T,M}
function signature_path!(
    out::Tensor{T,M},
    path::Vector{SVector{D,T}},
) where {D,T,M}

    @assert length(path) ≥ 2 "path must have at least 2 points"

    a = out
    b = similar(out)
    segment_tensor = similar(out)

    @inbounds begin
        # First segment
        Δ = path[2] - path[1]
        exp!(a, Δ)  # group-like, level-0 = 1

        # Remaining segments
        for i in 2:length(path)-1
            Δ = path[i+1] - path[i]
            exp!(segment_tensor, Δ)           # group-like
            mul_grouplike!(b, a, segment_tensor)
            a, b = b, a
        end
    end

    # There are nseg = length(path) - 1 segments.
    # After the loop:
    #   - if nseg is odd, result is already in `out`
    #   - if nseg is even, result is in `a` but `a !== out`
    if a !== out
        copy!(out, a)
    end

    return out
end

# Generic fallback for any other AbstractTensor backend:
# uses exp! + mul! without assuming group-like structure.
function signature_path!(
    out::AT,
    path::Vector{SVector{D,T}},
) where {D,T,AT<:AbstractTensor{T}}

    @assert length(path) ≥ 2 "path must have at least 2 points"

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
            mul!(b, a, segment_tensor)
            a, b = b, a
        end
    end

    if a !== out
        copy!(out, a)
    end

    return out
end
