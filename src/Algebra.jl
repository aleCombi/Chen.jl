module Algebra

using ..ChenSignatures: AbstractTensor, Tensor 
using LinearAlgebra

include("sparse_tensors.jl")
include("lyndon_basis.jl")

export SparseTensor, Word, shuffle_product, lyndon_words, build_L, project_to_lyndon

end