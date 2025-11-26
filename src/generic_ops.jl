# AbstractTensor is defined in Chen.jl

# --- Generic Exponential ---

"""
    exp!(out, X)

Compute the truncated power-series exponential:
    out = 1 + X + X^2/2! + ... + X^m/m!
Works for any tensor backend implementing _zero!, _write_unit!, add_scaled!, mul!, dim(), and level().
"""
function exp!(out::AbstractTensor{T}, X::AbstractTensor{T}) where {T}
    # Ensure dimensions match
    @assert dim(out)   == dim(X) "Dimension mismatch"
    @assert level(out) == level(X) "Level mismatch"

    # 1. Initialize out = 1
    _zero!(out)
    _write_unit!(out)    
    
    m = level(X)
    m == 0 && return out

    # 2. term = X (first term of series)
    term = similar(X)
    copy!(term, X)

    invfact = one(T)                # 1/1!
    add_scaled!(out, term, invfact) # out += X

    # 3. Accumulate higher powers
    tmp = similar(X)

    @inbounds for k in 2:m
        # tmp = term * X  (effectively X^k)
        mul!(tmp, term, X)
        
        # Swap buffers so 'term' holds the new result
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

# --- Operator Aliases ---

"""
    mul(a, b)
Generic multiplication wrapper (allocating).
"""
function mul(a::AbstractTensor, b::AbstractTensor)
    dest = similar(a, promote_type(eltype(a), eltype(b)))
    return mul!(dest, a, b)
end

const ⊗ = mul

# Note: We do NOT need `function exp! end` here anymore because we defined the method above.
# We still strictly require implementations for: mul!, log!
function log! end
function mul! end