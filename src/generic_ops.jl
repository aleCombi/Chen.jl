"""
    exp!(out, X)

Compute the truncated power-series exponential:
    out = 1 + X + X^2/2! + ... + X^m/m!
Works for any tensor backend implementing _zero!, _write_unit!, add_scaled!, mul!, dim(), and level().
"""
function exp!(out::AbstractTensor{T}, X::AbstractTensor{T}) where {T}
    @assert dim(out)   == dim(X) "Dimension mismatch"
    @assert level(out) == level(X) "Level mismatch"

    _zero!(out)
    _write_unit!(out)    
    
    m = level(X)
    m == 0 && return out

    term = similar(X)
    copy!(term, X)

    invfact = one(T)
    add_scaled!(out, term, invfact)

    tmp = similar(X)

    @inbounds for k in 2:m
        mul!(tmp, term, X)
        term, tmp = tmp, term
        invfact *= inv(T(k))
        add_scaled!(out, term, invfact)
    end
    return out
end

function exp(X::AT) where {AT<:AbstractTensor}
    out = similar(X)
    exp!(out, X)
    return out
end

"""
    mul(a, b)
Generic multiplication wrapper (allocating).
"""
function mul(a::AbstractTensor, b::AbstractTensor)
    dest = similar(a, promote_type(eltype(a), eltype(b)))
    return mul!(dest, a, b)
end

const ⊗ = mul

function log! end
function mul! end