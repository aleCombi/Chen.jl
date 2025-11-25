module Algebra

# Import AbstractTensor AND Tensor from parent so we can use them in conversions/lyndon
using ..Chen: AbstractTensor, Tensor 
using LinearAlgebra

include("sparse_tensors.jl")
include("lyndon_basis.jl")
include("conversions.jl")

export SparseTensor, Word, shuffle_product, resolvent, lyndon_words

end