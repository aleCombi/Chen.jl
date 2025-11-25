# src/generic_ops.jl

"""
    exp!(out, X)

Compute the truncated power-series exponential:
    out = 1 + X + X^2/2! + ... + X^m/m!
Works for any tensor backend implementing _zero!, _write_unit!, add_scaled!, mul!.
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
        # tmp <- term * X
        mul!(tmp, term, X)
        term, tmp = tmp, term

        invfact *= inv(T(k))
        add_scaled!(out, term, invfact)
    end
    return out
end

# Allocating wrapper
function exp(X::AT) where {AT<:AbstractTensor}
    out = similar(X)
    exp!(out, X)
    return out
end

# Operator aliases
function mul(a::AbstractTensor, b::AbstractTensor)
    dest = similar(a, promote_type(eltype(a), eltype(b)))
    return mul!(dest, a, b)
end

const ⊗ = mul