module PathSignatures

using StaticArrays
using LoopVectorization: @avx

include("tensor_series.jl")
export TensorSeries, signature_path

# ---------------- public API ----------------

function signature_path(path::Vector{SVector{D,T}}, m::Int) where {D,T}
    d = D
    a = TensorSeries{T}(d, m)
    b = TensorSeries{T}(d, m)
    segment_tensor = TensorSeries{T}(d, m)
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

include("symbolic_tensor_algebra.jl")
include("vol_signature.jl")
include("tensor_conversions.jl")

end # module PathSignatures
