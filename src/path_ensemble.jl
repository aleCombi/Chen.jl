using Random, StaticArrays

# === Data Structures ===

"""
    SVectorEnsemble{D,T}

Simple ensemble of paths where each path is Vector{SVector{D,T}}.
Optimized for flexibility and ease of use.
"""
struct SVectorEnsemble{D,T}
    paths::Vector{Vector{SVector{D,T}}}
    n_paths::Int
    n_steps::Int
    
    function SVectorEnsemble{D,T}(paths::Vector{Vector{SVector{D,T}}}) where {D,T}
        n_paths = length(paths)
        n_steps = isempty(paths) ? 0 : length(paths[1]) - 1
        new{D,T}(paths, n_paths, n_steps)
    end
end

"""
    ArrayEnsemble{T}

Array-based ensemble stored as (n_steps+1) × D × n_paths.
Optimized for batch operations and memory efficiency.
"""
struct ArrayEnsemble{T}
    data::Array{T,3}  # (n_steps+1) × D × n_paths
    n_paths::Int
    n_steps::Int
    dim::Int
    
    function ArrayEnsemble{T}(data::Array{T,3}) where {T}
        n_steps_plus_1, dim, n_paths = size(data)
        n_steps = n_steps_plus_1 - 1
        new{T}(data, n_paths, n_steps, dim)
    end
end

# === Brownian Motion Simulators ===

"""
    simulate_brownian_svector(::Type{SVector{D,T}}; n_paths=1000, T=1.0, n_steps=252, 
                              x0=zero(SVector{D,T}), rng=Random.GLOBAL_RNG)

Simulate D-dimensional Brownian motion using SVector representation.
Returns SVectorEnsemble{D,T}.

# Arguments
- First argument specifies the path point type (e.g., SVector{2,Float64})
- `n_paths`: Number of paths to simulate
- `T`: Final time
- `n_steps`: Number of time steps
- `x0`: Initial value as SVector{D,T}
- `rng`: Random number generator

# Example
```julia
# 2D Brownian motion
bm2d = simulate_brownian_svector(SVector{2,Float64}; n_paths=1000, T=1.0, n_steps=100)

# With custom initial condition
x0 = SVector(1.0, -0.5)
bm2d = simulate_brownian_svector(SVector{2,Float64}; x0=x0, n_paths=100)
```
"""
function simulate_brownian_svector(
    ::Type{SVector{D,T}};
    n_paths::Int = 1000,
    horizon::Real = 1.0,
    n_steps::Int = 252,
    x0::SVector{D,T} = zero(SVector{D,T}),
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {D,T<:AbstractFloat}
    
    dt = horizon / n_steps
    sqrt_dt = sqrt(dt)
    
    # Pre-allocate paths vector
    paths = Vector{Vector{SVector{D,T}}}(undef, n_paths)
    
    # Generate each path
    @inbounds for i in 1:n_paths
        path = Vector{SVector{D,T}}(undef, n_steps + 1)
        path[1] = x0
        
        # Generate increments for this path
        for j in 1:n_steps
            # Generate D-dimensional increment
            dW = SVector{D,T}(randn(rng, T) * sqrt_dt for _ in 1:D)
            path[j+1] = path[j] + dW
        end
        
        paths[i] = path
    end
    
    return SVectorEnsemble{D,T}(paths)
end

"""
    simulate_brownian_array(::Type{T}, D::Int; n_paths=1000, T=1.0, n_steps=252,
                           x0=zeros(T, D), rng=Random.GLOBAL_RNG)

Simulate D-dimensional Brownian motion using array representation.
Returns ArrayEnsemble{T}.

# Arguments
- First argument specifies the element type (e.g., Float64)
- `D`: Spatial dimension
- `n_paths`: Number of paths to simulate
- `T`: Final time
- `n_steps`: Number of time steps
- `x0`: Initial value as Vector{T} of length D
- `rng`: Random number generator

# Example
```julia
# 2D Brownian motion
bm2d = simulate_brownian_array(Float64, 2; n_paths=1000, T=1.0, n_steps=100)

# With custom initial condition
x0 = [1.0, -0.5]
bm2d = simulate_brownian_array(Float64, 2; x0=x0, n_paths=100)
```
"""
function simulate_brownian_array(
    ::Type{T}, D::Int;
    n_paths::Int = 1000,
    T_final::Real = 1.0,
    n_steps::Int = 252,
    x0::Vector{T} = zeros(T, D),
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T<:AbstractFloat}
    
    @assert length(x0) == D "Initial condition x0 must have length D=$D"
    
    dt = T_final / n_steps
    sqrt_dt = sqrt(dt)
    
    # Initialize array: (n_steps+1) × D × n_paths
    data = Array{T,3}(undef, n_steps + 1, D, n_paths)
    
    # Set initial conditions
    @inbounds for path_idx in 1:n_paths, d in 1:D
        data[1, d, path_idx] = x0[d]
    end
    
    # Generate increments and compute paths
    @inbounds for step in 1:n_steps
        for path_idx in 1:n_paths, d in 1:D
            increment = randn(rng, T) * sqrt_dt
            data[step + 1, d, path_idx] = data[step, d, path_idx] + increment
        end
    end
    
    return ArrayEnsemble{T}(data)
end

# === Convenience Functions ===

"""
    simulate_brownian_1d_svector(; kwargs...)

Convenience function for 1D Brownian motion using SVector representation.
"""
simulate_brownian_1d_svector(; kwargs...) = 
    simulate_brownian_svector(SVector{1,Float64}; kwargs...)

"""
    simulate_brownian_1d_array(; kwargs...)

Convenience function for 1D Brownian motion using array representation.
"""
simulate_brownian_1d_array(; kwargs...) = 
    simulate_brownian_array(Float64, 1; kwargs...)

# === Basic Accessors ===

"""
    get_path(ensemble::SVectorEnsemble, path_idx::Int)

Extract a single path from SVectorEnsemble.
"""
get_path(ensemble::SVectorEnsemble, path_idx::Int) = ensemble.paths[path_idx]

"""
    get_path(ensemble::ArrayEnsemble, path_idx::Int) -> Matrix

Extract a single path from ArrayEnsemble as (n_steps+1) × D matrix.
"""
function get_path(ensemble::ArrayEnsemble{T}, path_idx::Int) where {T}
    return ensemble.data[:, :, path_idx]
end

"""
    get_dimension(ensemble) -> Int

Get the spatial dimension.
"""
get_dimension(::SVectorEnsemble{D,T}) where {D,T} = D
get_dimension(ensemble::ArrayEnsemble) = ensemble.dim

# === Display Methods ===

function Base.show(io::IO, ensemble::SVectorEnsemble{D,T}) where {D,T}
    println(io, "SVectorEnsemble{$D,$T}:")
    println(io, "  Paths: $(ensemble.n_paths)")
    println(io, "  Time steps: $(ensemble.n_steps)")
    println(io, "  Spatial dimension: $D")
    
    if ensemble.n_paths > 0
        sample_point = ensemble.paths[1][1]
        println(io, "  Sample initial point: $sample_point")
    end
end

function Base.show(io::IO, ensemble::ArrayEnsemble{T}) where {T}
    println(io, "ArrayEnsemble{$T}:")
    println(io, "  Paths: $(ensemble.n_paths)")
    println(io, "  Time steps: $(ensemble.n_steps)")
    println(io, "  Spatial dimension: $(ensemble.dim)")
    println(io, "  Array size: $(size(ensemble.data))")
end

# === Iterator Interface ===

# SVectorEnsemble - iterate over paths directly
Base.iterate(ensemble::SVectorEnsemble) = iterate(ensemble.paths)
Base.iterate(ensemble::SVectorEnsemble, state) = iterate(ensemble.paths, state)
Base.length(ensemble::SVectorEnsemble) = ensemble.n_paths
Base.getindex(ensemble::SVectorEnsemble, i::Int) = ensemble.paths[i]

# ArrayEnsemble - iterate by extracting paths as matrices
struct ArrayEnsembleIterator{T}
    ensemble::ArrayEnsemble{T}
    current::Int
end

Base.iterate(ensemble::ArrayEnsemble) = 
    ensemble.n_paths > 0 ? (get_path(ensemble, 1), 2) : nothing

Base.iterate(ensemble::ArrayEnsemble, state::Int) = 
    state > ensemble.n_paths ? nothing : (get_path(ensemble, state), state + 1)

Base.length(ensemble::ArrayEnsemble) = ensemble.n_paths
Base.getindex(ensemble::ArrayEnsemble, i::Int) = get_path(ensemble, i)

# Export main types and functions
export SVectorEnsemble, ArrayEnsemble
export simulate_brownian_svector, simulate_brownian_array
export simulate_brownian_1d_svector, simulate_brownian_1d_array
export get_path, get_dimension