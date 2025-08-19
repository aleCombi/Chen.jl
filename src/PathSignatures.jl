module PathSignatures

using StaticArrays
using LoopVectorization: @avx, @turbo

include("dense_tensors.jl")
export Tensor, signature_path, AbstractTensor   

# ---------------- public API ----------------

function signature_path(::Type{AT}, path::Vector{SVector{D,T}}, m::Int) where {D,T, AT <: AbstractTensor{T}}
    out = AT(D,m)
    return signature_path!(out, path)
end

function signature_path!(out::AT, path::Vector{SVector{D,T}}) where {D,T, AT <: AbstractTensor{T}}
    d = D
    a = similar(out)
    b = similar(out)
    segment_tensor = similar(out)
    displacement = Vector{T}(undef, d)

    displacement .= path[2] - path[1] 
    exp!(a, displacement)

    for i in 2:length(path)-1
        displacement .= path[i+1] - path[i] 
        exp!(segment_tensor, displacement)
        mul!(b, a, segment_tensor)
        a, b = b, a
    end

    return a
end

include("sparse_tensors.jl")
include("vol_signature.jl")
include("tensor_conversions.jl")

end # module PathSignatures
