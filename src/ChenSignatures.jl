module ChenSignatures

using StaticArrays
using LinearAlgebra
using LoopVectorization

# 1. Base Type Definition
# Defined here to ensure visibility
abstract type AbstractTensor{T} end

# 2. Generic Operations
include("generic_ops.jl") 

# 3. Dense Engine
include("dense_tensors.jl")

# 4. Algebra Submodule
include("Algebra.jl") 
# IMPORT ALGEBRA SYMBOLS INTO CHEN SCOPE
# This is crucial so that 'SparseTensor' in Chen refers to the one in Algebra
using .Algebra: SparseTensor, Word, shuffle_product, lyndon_words, build_L, project_to_lyndon

# 5. Conversions
# Now conversions can safely refer to SparseTensor
include("conversions.jl")

# 6. Path Signature Logic
include("signatures.jl")

# 7. High-level User API
include("api.jl")

export sig, logsig, prepare
export Tensor, signature_path

end