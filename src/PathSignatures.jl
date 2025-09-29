module PathSignatures

include("tensors.jl")
include("dense_tensors.jl")
include("sparse_tensors.jl")
include("sparse_dense_conversions.jl")
include("path_ensemble.jl")  # consolidated version
include("signatures.jl")
include("gbm_simulation.jl")

export Tensor, signature_path, signature_path!, SparseTensor   
export SVectorEnsemble, ArrayEnsemble
export simulate_brownian_svector, simulate_brownian_array
export simulate_brownian_1d_svector, simulate_brownian_1d_array
export get_path, get_dimension
export batch_signatures!, batch_signatures, prepare_signature_outputs

end # module PathSignatures