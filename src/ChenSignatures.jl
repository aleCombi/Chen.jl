module ChenSignatures

using StaticArrays
using LinearAlgebra
using LoopVectorization

abstract type AbstractTensor{T} end

include("generic_ops.jl")
include("sparse_tensors.jl")
include("dense_tensors.jl") 
include("lyndon_basis.jl")
include("conversions.jl")
include("signatures.jl")

export sig, logsig, prepare
export Tensor, signature_path
export SparseTensor, Word, shuffle_product, lyndon_words, build_L, project_to_lyndon

end