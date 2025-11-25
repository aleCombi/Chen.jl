module Chen

using StaticArrays
using LinearAlgebra
using LoopVectorization

# 1. Base Type
abstract type AbstractTensor{T} end

# 2. Generic Algorithms (Rename your old tensors.jl to generic_ops.jl)
#    This defines generic exp! for AbstractTensor
include("generic_ops.jl") 

# 3. Dense Engine (Defines Tensor)
#    MUST come before Algebra so Algebra can see 'Tensor'
include("dense_tensors.jl")

# 4. Algebra Submodule (Defines SparseTensor, uses Tensor)
include("Algebra.jl") 

# 5. User API
include("signatures.jl")
include("api.jl")

export sig, logsig, prepare
export Tensor, signature_path

end