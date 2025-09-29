abstract type AbstractTensor{T} end

"""
    exp!(out, X)

Compute the truncated power-series exponential:
    out = X + X^2/2! + ... + X^m/m!
(Level-0 unit is implicit and not stored; backends control truncation via `level`.)
Works for any `Tensor` backend implementing the small interface.
"""
function exp!(out::AbstractTensor{T}, X::AbstractTensor{T}) where {T}
    @assert dim(out)   == dim(X)
    @assert level(out) == level(X)

    _zero!(out)
    _write_unit!(out)    
    m = level(X)
    m == 0 && return out

    # term = X^1
    term = similar(X)
    copy!(term, X)

    invfact = one(T)                # 1/1!
    add_scaled!(out, term, invfact) # add X

    # tmp buffer for powers
    tmp = similar(X)

    @inbounds for k in 2:m
        # tmp <- term * X   (do not assume alias-safety)
        mul!(tmp, term, X)
        term, tmp = tmp, term

        invfact *= inv(T(k))        # update 1/k!
        add_scaled!(out, term, invfact)
    end
    return out
end

# Allocating wrapper (works for any backend)
function Base.exp(X::AT) where {AT<:AbstractTensor}
    out = similar(X)
    exp!(out, X)
    return out
end