# AbstractTensor is defined in Chen.jl

"""
    mul(a, b)
Generic multiplication wrapper (allocating).
"""
function mul(a::AbstractTensor, b::AbstractTensor)
    dest = similar(a, promote_type(eltype(a), eltype(b)))
    return mul!(dest, a, b)
end

const ⊗ = mul

# Fallback declarations
function exp! end
function mul! end
function log! end