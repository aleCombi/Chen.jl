# ---------------- public API ----------------

# In src/signatures.jl, inside module Chen

function signature_path(
    ::Type{AT},
    path::Vector{SVector{D,T}},
    m::Int,
) where {D,T,AT<:AbstractTensor{T}}

    @assert length(path) ≥ 2 "path must have at least 2 points"

    out = AT(D, m)
    signature_path!(out, path)
    return out
end


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
            mul_grouplike!(b, a, segment_tensor)
            a, b = b, a
        end
    end

    # There are nseg = length(path) - 1 segments.
    # After the loop:
    #   - if nseg is odd, result is already in `out`
    #   - if nseg is even, result is in `b`
    # The pointer check captures exactly that.
    if a !== out
        copy!(out, a)
    end

    return out
end
