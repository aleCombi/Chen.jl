using LinearAlgebra
using StaticArrays

export sig, prepare, logsig, signature_path, SignatureWorkspace, BasisCache

# ============================================================================
# Workspace Preallocation
# ============================================================================

"""
    SignatureWorkspace{T,D,M}

Preallocated workspace for computing path signatures without allocations.

This workspace contains scratch buffers used internally by the Horner update scheme.
Reusing a workspace across multiple signature computations eliminates memory allocations
in the hot path, which is beneficial for batch processing or moving window applications.

# Type Parameters
- `T`: Element type (e.g., `Float64`, `Float32`)
- `D`: Path dimension (number of coordinates per point)
- `M`: Truncation level (maximum signature level to compute)

# Fields
- `B1::Vector{T}`: First scratch buffer of length `D^(M-1)`
- `B2::Vector{T}`: Second scratch buffer of length `D^(M-1)`

# Example
```julia
# Create workspace for dimension 3, level 4
ws = SignatureWorkspace{Float64, 3, 4}()

# Reuse workspace for multiple paths
for path in paths
    out = Tensor{Float64, 3, 4}()
    signature_path!(out, path, ws)
    # process out...
end
```

See also: [`signature_path!`](@ref)
"""
struct SignatureWorkspace{T,D,M}
    B1::Vector{T}
    B2::Vector{T}
end

"""
    SignatureWorkspace{T,D,M}()

Create a preallocated workspace for signature computation.

Allocates scratch buffers of size `D^(M-1)` for the Horner update scheme.
"""
function SignatureWorkspace{T,D,M}() where {T,D,M}
    max_len = M > 1 ? D^(M-1) : 1
    return SignatureWorkspace{T,D,M}(
        Vector{T}(undef, max_len),
        Vector{T}(undef, max_len)
    )
end

# ============================================================================
# Public API Functions
# ============================================================================

"""
    sig(path::AbstractMatrix, m::Int) -> Vector

Compute the truncated path signature up to level `m`.

The path signature is a graded feature vector that characterizes the path's geometry.
It consists of iterated integrals computed recursively using Chen's identity.

# Arguments
- `path::AbstractMatrix{T}`: `N×d` matrix where `N ≥ 2` is the number of points and
  `d ≥ 1` is the dimension. Each row represents a point in d-dimensional space.
- `m::Int`: Truncation level (`m ≥ 1`). The signature will include levels 1 through `m`.

# Returns
- `Vector{T}`: Flattened coefficient vector of length `d + d² + ... + dᵐ` containing
  signature coefficients ordered by level (all level-1 terms, then level-2, etc.).

# Computational Complexity
- Time: O(N · dᵐ⁺¹) where N is path length
- Space: O(dᵐ⁺¹) for signature storage

# Examples
```julia
# 2D path with 100 points, signature up to level 3
path = randn(100, 2)
s = sig(path, 3)
length(s)  # 2 + 4 + 8 = 14

# 3D path, level 4
path = randn(50, 3)
s = sig(path, 4)
length(s)  # 3 + 9 + 27 + 81 = 120
```

# Notes
- This function allocates scratch buffers internally. For repeated calls, consider
  using `signature_path!` with a preallocated [`SignatureWorkspace`](@ref).
- For log-signature (more compact representation), see [`logsig`](@ref).

# References
K.T. Chen (1957). "Integration of paths, geometric invariants and a generalized
Baker-Hausdorff formula."

See also: [`logsig`](@ref), [`prepare`](@ref), [`signature_path!`](@ref)
"""
function sig(path::AbstractMatrix{T}, m::Int) where T
    N, D = size(path)

    # Input validation
    N >= 2 || throw(ArgumentError("Path must have at least 2 points, got N=$N"))
    D >= 1 || throw(ArgumentError("Path dimension must be at least 1, got D=$D"))
    m >= 1 || throw(ArgumentError("Signature level must be at least 1, got m=$m"))

    out = Tensor{T, D, m}()
    signature_path!(out, path)
    return _flatten_tensor(out)
end

"""
    BasisCache{T}

Cached Lyndon basis data for efficient log-signature computation.

This structure stores precomputed data for projecting signatures onto the Lyndon basis,
which provides a minimal free-Lie algebra representation of the log-signature.

# Fields
- `d::Int`: Path dimension
- `m::Int`: Truncation level
- `lynds::Vector{Word}`: Lyndon words up to length `m`
- `L::Matrix{T}`: Lower-triangular projection matrix

# Notes
Create instances using [`prepare`](@ref), not directly.

See also: [`prepare`](@ref), [`logsig`](@ref)
"""
struct BasisCache{T}
    d::Int
    m::Int
    lynds::Vector{Word}
    L::Matrix{T}
end

"""
    prepare(d::Int, m::Int) -> BasisCache

Precompute Lyndon basis and projection matrix for log-signature computation.

This function builds the data structures needed for [`logsig`](@ref). The Lyndon basis
provides a Hall basis for the free Lie algebra, giving a minimal representation of the
log-signature with dimension `dim_Lie(d,m) ≪ d + d² + ... + dᵐ`.

# Arguments
- `d::Int`: Path dimension (`d ≥ 1`)
- `m::Int`: Truncation level (`m ≥ 1`)

# Returns
- `BasisCache`: Opaque structure containing basis information. Pass to [`logsig`](@ref).

# Performance Notes
- This computation can be expensive for large `d` and `m` (involves symbolic shuffles).
- Cache the result and reuse for multiple paths with the same `(d, m)`.
- Time complexity: O(L²) where L is the number of Lyndon words.

# Examples
```julia
# Precompute basis for 3D paths, level 5
basis = prepare(3, 5)

# Reuse for multiple paths
for path in paths
    ls = logsig(path, basis)
    # process ls...
end
```

# References
Reutenauer (1993). "Free Lie Algebras." Oxford University Press.

See also: [`logsig`](@ref), [`BasisCache`](@ref)
"""
function prepare(d::Int, m::Int)
    # Input validation
    d >= 1 || throw(ArgumentError("Path dimension must be at least 1, got d=$d"))
    m >= 1 || throw(ArgumentError("Signature level must be at least 1, got m=$m"))

    lynds, L, _ = build_L(d, m)
    return BasisCache(d, m, lynds, L)
end

"""
    logsig(path::AbstractMatrix, basis::BasisCache) -> Vector

Compute the log-signature of a path projected onto the Lyndon basis.

The log-signature is the logarithm (in the tensor algebra) of the signature, and provides
a more compact representation. When projected onto the Lyndon basis, it gives coefficients
in the free Lie algebra, which is much smaller than the full tensor algebra.

# Arguments
- `path::AbstractMatrix{T}`: `N×d` matrix where `N ≥ 2` is the number of points.
  Dimension `d` must match the basis.
- `basis::BasisCache`: Precomputed basis from [`prepare(d, m)`](@ref)

# Returns
- `Vector{T}`: Log-signature coefficients in Lyndon basis. Length equals number of
  Lyndon words up to length `m`.

# Computational Complexity
- Time: O(N · dᵐ⁺¹ + L²) where L is number of Lyndon words
- Space: O(dᵐ⁺¹) for intermediate signature computation

# Examples
```julia
# Setup
path = randn(100, 3)
basis = prepare(3, 4)

# Compute log-signature
ls = logsig(path, basis)

# Log-signature is much more compact than signature
println("Log-signature size: ", length(ls))        # Depends on Lyndon words
println("Full signature size: ", 3 + 9 + 27 + 81)  # 120
```

# Notes
- Requires precomputed `basis` from [`prepare`](@ref).
- More compact than [`sig`](@ref) but requires additional preprocessing.
- Best suited when you need many log-signatures with the same `(d, m)`.

# References
Lyons, Caruana, Lévy (2007). "Differential equations driven by rough paths."
Lecture Notes in Mathematics.

See also: [`sig`](@ref), [`prepare`](@ref), [`BasisCache`](@ref)
"""
function logsig(path::AbstractMatrix{T}, basis::BasisCache) where T
    N, D = size(path)

    # Input validation
    N >= 2 || throw(ArgumentError("Path must have at least 2 points, got N=$N"))
    D == basis.d || throw(ArgumentError(
        "Dimension mismatch: path has dimension $D but basis expects $(basis.d)"
    ))

    sig_tensor = Tensor{T, basis.d, basis.m}()
    signature_path!(sig_tensor, path)

    log_tensor = ChenSignatures.log(sig_tensor)
    return project_to_lyndon(log_tensor, basis.lynds, basis.L)
end

# ============================================================================
# Lower-level API (for advanced users)
# ============================================================================

"""
    signature_path(::Type{Tensor{T}}, path, m::Int) -> Tensor{T,D,m}

Compute path signature and return as a `Tensor` (not flattened).

This is a lower-level function that returns the signature in tensor form rather than
as a flattened vector. Most users should use [`sig`](@ref) instead.

# Arguments
- `::Type{Tensor{T}}`: Element type for the output tensor
- `path`: Path as `AbstractMatrix{T}` or `AbstractVector{SVector{D,T}}`
- `m::Int`: Truncation level

# Returns
- `Tensor{T,D,m}`: Signature tensor with graded structure preserved

See also: [`sig`](@ref), [`signature_path!`](@ref)
"""
function signature_path(::Type{Tensor{T}}, path, m::Int) where T
    D = _get_dim(path)
    out = Tensor{T, D, m}()
    signature_path!(out, path)
    return out
end

_get_dim(path::AbstractMatrix) = size(path, 2)
_get_dim(path::AbstractVector{SVector{D, T}}) where {D, T} = D

"""
    signature_path!(out::Tensor{T,D,M}, path) -> Tensor{T,D,M}
    signature_path!(out::Tensor{T,D,M}, path, workspace::SignatureWorkspace{T,D,M}) -> Tensor{T,D,M}

Compute path signature in-place with optional workspace preallocation.

# Arguments
- `out::Tensor{T,D,M}`: Preallocated output tensor (will be overwritten)
- `path`: Path as `N×D` matrix or vector of `SVector{D,T}`
- `workspace::SignatureWorkspace{T,D,M}` (optional): Preallocated scratch buffers

# Returns
- The modified `out` tensor containing the signature

# Performance Notes
- **Without workspace**: Allocates `O(D^(M-1))` scratch space each call. Convenient for
  single computations but inefficient for repeated calls.
- **With workspace**: Zero-allocation hot path. Use this for batch processing, moving
  windows, or any scenario with repeated signature computations.

# Examples
```julia
# Simple usage (allocates internally)
path = randn(100, 3)
out = Tensor{Float64, 3, 4}()
signature_path!(out, path)

# High-performance usage (zero allocation)
ws = SignatureWorkspace{Float64, 3, 4}()
for path in paths
    signature_path!(out, path, ws)  # No allocations in hot path
    # process out...
end
```

See also: [`SignatureWorkspace`](@ref), [`sig`](@ref)
"""
function signature_path! end

# Allocating version (for user convenience)
function signature_path!(out::Tensor{T,D,M}, path::AbstractMatrix{T}) where {T,D,M}
    N = size(path, 1)
    N >= 2 || throw(ArgumentError("Path must have at least 2 points, got N=$N"))

    _reset_tensor!(out)
    B1, B2 = _alloc_scratch(T, D, M)

    @inbounds for i in 1:N-1
        val = ntuple(j -> path[i+1, j] - path[i, j], Val(D))
        z = SVector{D, T}(val)
        update_signature_horner!(out, z, B1, B2)
    end
    return out
end

function signature_path!(out::Tensor{T,D,M}, path::AbstractVector{SVector{D,T}}) where {T,D,M}
    N = length(path)
    N >= 2 || throw(ArgumentError("Path must have at least 2 points, got N=$N"))

    _reset_tensor!(out)
    B1, B2 = _alloc_scratch(T, D, M)

    @inbounds for i in 1:N-1
        z = path[i+1] - path[i]
        update_signature_horner!(out, z, B1, B2)
    end
    return out
end

# Workspace-based versions (zero allocation hot path)
function signature_path!(
    out::Tensor{T,D,M},
    path::AbstractMatrix{T},
    ws::SignatureWorkspace{T,D,M}
) where {T,D,M}
    N = size(path, 1)
    N >= 2 || throw(ArgumentError("Path must have at least 2 points, got N=$N"))

    _reset_tensor!(out)

    @inbounds for i in 1:N-1
        val = ntuple(j -> path[i+1, j] - path[i, j], Val(D))
        z = SVector{D, T}(val)
        update_signature_horner!(out, z, ws.B1, ws.B2)
    end
    return out
end

function signature_path!(
    out::Tensor{T,D,M},
    path::AbstractVector{SVector{D,T}},
    ws::SignatureWorkspace{T,D,M}
) where {T,D,M}
    N = length(path)
    N >= 2 || throw(ArgumentError("Path must have at least 2 points, got N=$N"))

    _reset_tensor!(out)

    @inbounds for i in 1:N-1
        z = path[i+1] - path[i]
        update_signature_horner!(out, z, ws.B1, ws.B2)
    end
    return out
end

@inline function _reset_tensor!(out::Tensor{T}) where T
    fill!(out.coeffs, zero(T))
    ChenSignatures._write_unit!(out)
end

@inline function _alloc_scratch(::Type{T}, D::Int, M::Int) where T
    max_len = M > 1 ? D^(M-1) : 1
    return Vector{T}(undef, max_len), Vector{T}(undef, max_len)
end

function _flatten_tensor(t::Tensor{T,D,M}) where {T,D,M}
    total_len = t.offsets[end] - t.offsets[2] 
    out = Vector{T}(undef, total_len)
    
    current_idx = 1
    for k in 1:M
        start_offset = t.offsets[k+1]
        len = D^k
        copyto!(out, current_idx, t.coeffs, start_offset + 1, len)
        current_idx += len
    end
    return out
end