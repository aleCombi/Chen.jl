using Random, Statistics, StaticArrays

"""
    PathEnsemble{D,T}

A structure containing an ensemble of paths with metadata.
Each path is represented as Vector{SVector{D,T}}.

Fields:
- `paths`: Vector of paths, where each path is Vector{SVector{D,T}}
- `times`: Vector of time points (supports non-uniform grids)  
- `n_paths`: Number of paths in the ensemble
- `n_steps`: Number of time steps per path
- `dt`: Time step size for uniform grids, NaN for non-uniform grids
"""
struct PathEnsemble{D,T<:AbstractFloat}
    paths::Vector{Vector{SVector{D,T}}}
    times::Vector{T}
    n_paths::Int
    n_steps::Int
    dt::T  # NaN for non-uniform grids
    
    function PathEnsemble{D,T}(paths::Vector{Vector{SVector{D,T}}}, times::Vector{T}) where {D,T<:AbstractFloat}
        n_paths = length(paths)
        n_steps = length(times) - 1
        
        # Check if grid is uniform
        if length(times) > 1
            dts = diff(times)
            dt = isapprox(minimum(dts), maximum(dts), rtol=1e-10) ? dts[1] : T(NaN)
        else
            dt = T(NaN)
        end
        
        new{D,T}(paths, times, n_paths, n_steps, dt)
    end
end

# Convenience constructor that infers D,T
function PathEnsemble(paths::Vector{Vector{SVector{D,T}}}, times::Vector{T}) where {D,T}
    return PathEnsemble{D,T}(paths, times)
end

"""
    simulate_brownian(::Type{SVector{D,T}}; n_paths=1000, T=1.0, n_steps=252, 
                      x0=zero(SVector{D,T}), times=nothing, rng=Random.GLOBAL_RNG)

Simulate D-dimensional Brownian motion paths using SVector representation.

Arguments:
- First argument specifies the path point type (e.g., SVector{2,Float64})
- `n_paths`: Number of paths to simulate
- `T`: Final time (ignored if `times` is provided)  
- `n_steps`: Number of time steps (ignored if `times` is provided)
- `x0`: Initial value as SVector{D,T}
- `times`: Custom time grid (overrides T and n_steps if provided)
- `rng`: Random number generator

Returns:
- `PathEnsemble{D,T}` containing the simulated paths

Example:
```julia
# 1D Brownian motion
bm1d = simulate_brownian(SVector{1,Float64}; n_paths=1000, T=1.0, n_steps=100)

# 2D Brownian motion  
bm2d = simulate_brownian(SVector{2,Float64}; n_paths=500, T=2.0, n_steps=200)

# With custom initial condition
x0 = SVector(1.0, -0.5)
bm2d = simulate_brownian(SVector{2,Float64}; x0=x0, n_paths=100)
```
"""
function simulate_brownian(
    ::Type{SVector{D,T}};
    n_paths::Int = 1000,
    T::Real = 1.0,
    n_steps::Int = 252,
    x0::SVector{D,T} = zero(SVector{D,T}),
    times::Vector{<:Real} = Vector{T}(range(zero(T), T, length=n_steps+1)),
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {D,T<:AbstractFloat}
    
    times_T = Vector{T}(times)
    n_steps = length(times_T) - 1
    
    # Pre-allocate paths vector
    paths = Vector{Vector{SVector{D,T}}}(undef, n_paths)
    
    # Generate each path
    @inbounds for i in 1:n_paths
        path = Vector{SVector{D,T}}(undef, n_steps + 1)
        path[1] = x0
        
        # Generate increments for this path
        if n_steps > 0
            dt_vec = diff(times_T)
            
            for j in 1:n_steps
                # Generate D-dimensional increment
                sqrt_dt = sqrt(dt_vec[j])
                dW = SVector{D,T}(randn(rng, T) * sqrt_dt for _ in 1:D)
                path[j+1] = path[j] + dW
            end
        end
        
        paths[i] = path
    end
    
    return PathEnsemble{D,T}(paths, times_T)
end

"""
    simulate_brownian_matrix(::Type{SVector{D,T}}; n_paths=1000, T=1.0, n_steps=252,
                            x0=zero(SVector{D,T}), times=nothing, rng=Random.GLOBAL_RNG)

Alternative implementation using a matrix-based approach for potentially better performance.
Stores paths as a (n_steps+1) × (D*n_paths) matrix internally, then converts to Vector{Vector{SVector{D,T}}}.

This can be faster for large ensembles due to better memory layout and vectorization,
but uses more memory temporarily.
"""
function simulate_brownian_matrix(
    ::Type{SVector{D,T}};
    n_paths::Int = 1000,
    T::Real = 1.0,
    n_steps::Int = 252,
    x0::SVector{D,T} = zero(SVector{D,T}),
    times::Vector{<:Real} = Vector{T}(range(zero(T), T, length=n_steps+1)),
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {D,T<:AbstractFloat}
    
    times_T = Vector{T}(times)
    n_steps = length(times_T) - 1
    
    if n_steps == 0
        paths = [SVector{D,T}[x0] for _ in 1:n_paths]
        return PathEnsemble{D,T}(paths, times_T)
    end
    
    # Work with matrix: rows = time steps, cols = flattened (path, dimension)
    # Column layout: [path1_dim1, path1_dim2, ..., path1_dimD, path2_dim1, ...]
    n_cols = n_paths * D
    path_matrix = Matrix{T}(undef, n_steps + 1, n_cols)
    
    # Set initial conditions
    @inbounds for path_idx in 1:n_paths
        for d in 1:D
            col_idx = (path_idx - 1) * D + d
            path_matrix[1, col_idx] = x0[d]
        end
    end
    
    # Generate all increments at once
    dt_vec = diff(times_T)
    increments = randn(rng, n_steps, n_cols)
    
    # Scale increments and compute paths
    @inbounds for step in 1:n_steps
        sqrt_dt = sqrt(dt_vec[step])
        for col in 1:n_cols
            increment = increments[step, col] * sqrt_dt
            path_matrix[step + 1, col] = path_matrix[step, col] + increment
        end
    end
    
    # Convert back to Vector{Vector{SVector{D,T}}} format
    paths = Vector{Vector{SVector{D,T}}}(undef, n_paths)
    @inbounds for path_idx in 1:n_paths
        path = Vector{SVector{D,T}}(undef, n_steps + 1)
        
        for time_idx in 1:(n_steps + 1)
            coords = ntuple(d -> path_matrix[time_idx, (path_idx - 1) * D + d], D)
            path[time_idx] = SVector{D,T}(coords)
        end
        
        paths[path_idx] = path
    end
    
    return PathEnsemble{D,T}(paths, times_T)
end

# Convenience methods for 1D case (common usage)
"""
    simulate_brownian_1d(; kwargs...)

Convenience function for 1D Brownian motion. Equivalent to:
`simulate_brownian(SVector{1,Float64}; kwargs...)`
"""
simulate_brownian_1d(; kwargs...) = simulate_brownian(SVector{1,Float64}; kwargs...)

"""
    simulate_brownian_1d_matrix(; kwargs...)

Matrix-based version for 1D Brownian motion.
"""
simulate_brownian_1d_matrix(; kwargs...) = simulate_brownian_matrix(SVector{1,Float64}; kwargs...)

# Accessor methods
"""
    get_path(ensemble::PathEnsemble, path_idx::Int)

Extract a single path from PathEnsemble structure.
Returns Vector{SVector{D,T}}.
"""
get_path(ensemble::PathEnsemble, path_idx::Int) = ensemble.paths[path_idx]

"""
    get_times(ensemble::PathEnsemble)

Get the time grid.
"""
get_times(ensemble::PathEnsemble) = ensemble.times

"""
    get_endpoints(ensemble::PathEnsemble)

Get the final values of all paths as Vector{SVector{D,T}}.
"""
function get_endpoints(ensemble::PathEnsemble{D,T}) where {D,T}
    return SVector{D,T}[path[end] for path in ensemble.paths]
end

"""
    get_dimension(ensemble::PathEnsemble{D,T}) where {D,T}

Get the spatial dimension D.
"""
get_dimension(::PathEnsemble{D,T}) where {D,T} = D

"""
    is_uniform_grid(ensemble::PathEnsemble)

Check if the time grid is uniform.
"""
is_uniform_grid(ensemble::PathEnsemble) = !isnan(ensemble.dt)

"""
    path_to_matrix(path::Vector{SVector{D,T}})

Convert a single path to a matrix where columns are dimensions, rows are time steps.
Useful for plotting or interfacing with other libraries.
"""
function path_to_matrix(path::Vector{SVector{D,T}}) where {D,T}
    n_steps = length(path)
    mat = Matrix{T}(undef, n_steps, D)
    @inbounds for i in 1:n_steps, d in 1:D
        mat[i, d] = path[i][d]
    end
    return mat
end

"""
    ensemble_to_array(ensemble::PathEnsemble{D,T})

Convert all paths to a 3D array of size (n_steps+1, D, n_paths).
"""
function ensemble_to_array(ensemble::PathEnsemble{D,T}) where {D,T}
    n_steps_plus_1 = ensemble.n_steps + 1
    result = Array{T,3}(undef, n_steps_plus_1, D, ensemble.n_paths)
    
    @inbounds for (path_idx, path) in enumerate(ensemble.paths)
        for (time_idx, point) in enumerate(path)
            for d in 1:D
                result[time_idx, d, path_idx] = point[d]
            end
        end
    end
    
    return result
end

# Display method
function Base.show(io::IO, ensemble::PathEnsemble{D,T}) where {D,T}
    println(io, "PathEnsemble{$D,$T}:")
    println(io, "  Paths: $(ensemble.n_paths)")
    println(io, "  Time steps: $(ensemble.n_steps)")  
    println(io, "  Spatial dimension: $D")
    println(io, "  Time range: [$(first(ensemble.times)), $(last(ensemble.times))]")
    if is_uniform_grid(ensemble)
        println(io, "  Uniform dt: $(ensemble.dt)")
    else
        println(io, "  Non-uniform time grid")
    end
    
    # Show a sample path point for context
    if ensemble.n_paths > 0 && ensemble.n_steps >= 0
        sample_point = ensemble.paths[1][1]
        println(io, "  Sample initial point: $sample_point")
    end
end

# Iterator interface - iterate over paths
Base.iterate(ensemble::PathEnsemble) = iterate(ensemble.paths)
Base.iterate(ensemble::PathEnsemble, state) = iterate(ensemble.paths, state)
Base.length(ensemble::PathEnsemble) = ensemble.n_paths
Base.getindex(ensemble::PathEnsemble, i::Int) = ensemble.paths[i]