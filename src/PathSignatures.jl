module Chen

include("tensors.jl")
include("dense_tensors.jl")
include("sparse_tensors.jl")
include("sparse_dense_conversions.jl")
include("signatures.jl")
include("lyndon_basis.jl")
export Tensor, signature_path, SparseTensor

end # module Chen