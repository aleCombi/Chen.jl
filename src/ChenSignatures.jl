module ChenSignatures

using StaticArrays
using LinearAlgebra

abstract type AbstractTensor{T} end

include("generic_ops.jl")
include("sparse_tensors.jl")
include("dense_tensors.jl")
include("lyndon_basis.jl")
include("conversions.jl")
include("signatures.jl")

using ChainRulesCore
using Enzyme
include("chain_rules.jl")

# Optional GPU support via KernelAbstractions
using KernelAbstractions
include("gpu_kernels.jl")

# ============================================================================
# Public API Exports
# ============================================================================
#
# Primary user-facing functions for computing path signatures:
export sig          # Compute truncated path signature (returns flattened vector)
export logsig       # Compute log-signature projected onto Lyndon basis
export prepare      # Precompute Lyndon basis for log-signature computations
export rolling_sig  # Compute signatures over rolling windows of a time series path
#
# Core types and lower-level API:
export Tensor              # Dense tensor algebra representation
export signature_path      # Lower-level signature computation (returns Tensor)
export SignatureWorkspace  # Preallocated workspace for zero-allocation hot paths
export BasisCache          # Cached Lyndon basis data for logsig
#
# GPU acceleration (requires KernelAbstractions.jl + GPU backend):
export sig_batch_gpu       # GPU-accelerated batch signature computation

# Note: SparseTensor, Word, shuffle_product, lyndon_words, build_L, and
# project_to_lyndon are internal implementation details and not exported.
# Advanced users can access them via ChenSignatures.SparseTensor, etc.

end