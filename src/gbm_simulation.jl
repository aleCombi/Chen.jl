# === Geometric Brownian Motion Simulators ===

"""
    simulate_gbm_svector(::Type{SVector{D,T}}; n_paths=1000, T=1.0, n_steps=252,
                         x0=ones(SVector{D,T}), μ=zeros(SVector{D,T}), σ=ones(SVector{D,T}),
                         rng=Random.GLOBAL_RNG)

Simulate D-dimensional geometric Brownian motion using SVector representation.
Returns SVectorEnsemble{D,T}.

The SDE is: dXₜ = diag(μ)Xₜdt + diag(σ)XₜdWₜ
Solution: Xₜ = X₀ ∘ exp((μ - σ²/2)t + σ∘Wₜ) where ∘ denotes elementwise operations.

# Arguments
- First argument specifies the path point type (e.g., SVector{2,Float64})
- `n_paths`: Number of paths to simulate
- `horizon`: Final time
- `n_steps`: Number of time steps
- `x0`: Initial value as SVector{D,T}
- `μ`: Drift parameter as SVector{D,T}
- `σ`: Volatility parameter as SVector{D,T}
- `rng`: Random number generator

# Example
```julia
# 2D GBM with different parameters per dimension
μ = SVector(0.05, 0.08)
σ = SVector(0.2, 0.3)
x0 = SVector(100.0, 50.0)
gbm2d = simulate_gbm_svector(SVector{2,Float64}; x0=x0, μ=μ, σ=σ, n_paths=1000)
```
"""
function simulate_gbm_svector(
    ::Type{SVector{D,T}};
    n_paths::Int = 1000,
    horizon::Real = 1.0,
    n_steps::Int = 252,
    x0::SVector{D,T} = ones(SVector{D,T}),
    μ::SVector{D,T} = zero(SVector{D,T}),
    σ::SVector{D,T} = ones(SVector{D,T}),
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {D,T<:AbstractFloat}
    
    dt = horizon / n_steps
    sqrt_dt = sqrt(dt)
    
    # Precompute drift term: (μ - σ²/2)dt
    drift = @. (μ - σ^2 / 2) * dt
    vol_sqrt_dt = σ .* sqrt_dt
    
    # Pre-allocate paths vector
    paths = Vector{Vector{SVector{D,T}}}(undef, n_paths)
    
    # Generate each path
    @inbounds for i in 1:n_paths
        path = Vector{SVector{D,T}}(undef, n_steps + 1)
        path[1] = x0
        
        # Generate increments for this path
        for j in 1:n_steps
            # Generate D-dimensional increment
            dW = SVector{D,T}(randn(rng, T) for _ in 1:D)
            
            # GBM update: X[j+1] = X[j] * exp(drift + σ*sqrt(dt)*dW)
            log_increment = drift .+ vol_sqrt_dt .* dW
            path[j+1] = path[j] .* exp.(log_increment)
        end
        
        paths[i] = path
    end
    
    return SVectorEnsemble{D,T}(paths)
end

"""
    simulate_gbm_array(::Type{T}, D::Int; n_paths=1000, T=1.0, n_steps=252,
                       x0=ones(T, D), μ=zeros(T, D), σ=ones(T, D),
                       rng=Random.GLOBAL_RNG)

Simulate D-dimensional geometric Brownian motion using array representation.
Returns ArrayEnsemble{T}.

The SDE is: dXₜ = diag(μ)Xₜdt + diag(σ)XₜdWₜ

# Arguments
- First argument specifies the element type (e.g., Float64)
- `D`: Spatial dimension
- `n_paths`: Number of paths to simulate
- `horizon`: Final time
- `n_steps`: Number of time steps
- `x0`: Initial value as Vector{T} of length D
- `μ`: Drift parameter as Vector{T} of length D
- `σ`: Volatility parameter as Vector{T} of length D
- `rng`: Random number generator

# Example
```julia
# 2D GBM with different parameters per dimension
μ = [0.05, 0.08]
σ = [0.2, 0.3]
x0 = [100.0, 50.0]
gbm2d = simulate_gbm_array(Float64, 2; x0=x0, μ=μ, σ=σ, n_paths=1000)
```
"""
function simulate_gbm_array(
    ::Type{T}, D::Int;
    n_paths::Int = 1000,
    horizon::Real = 1.0,
    n_steps::Int = 252,
    x0::Vector{T} = ones(T, D),
    μ::Vector{T} = zeros(T, D),
    σ::Vector{T} = ones(T, D),
    rng::AbstractRNG = Random.GLOBAL_RNG
) where {T<:AbstractFloat}
    
    @assert length(x0) == D "Initial condition x0 must have length D=$D"
    @assert length(μ) == D "Drift μ must have length D=$D"
    @assert length(σ) == D "Volatility σ must have length D=$D"
    
    dt = horizon / n_steps
    sqrt_dt = sqrt(dt)
    
    # Precompute drift term: (μ - σ²/2)dt
    drift = @. (μ - σ^2 / 2) * dt
    vol_sqrt_dt = σ .* sqrt_dt
    
    # Initialize array: (n_steps+1) × D × n_paths
    data = Array{T,3}(undef, n_steps + 1, D, n_paths)
    
    # Set initial conditions
    @inbounds for path_idx in 1:n_paths, d in 1:D
        data[1, d, path_idx] = x0[d]
    end
    
    # Generate increments and compute paths
    @inbounds for step in 1:n_steps
        for path_idx in 1:n_paths, d in 1:D
            dW = randn(rng, T)
            log_increment = drift[d] + vol_sqrt_dt[d] * dW
            data[step + 1, d, path_idx] = data[step, d, path_idx] * exp(log_increment)
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

"""
    simulate_gbm_1d_svector(; kwargs...)

Convenience function for 1D geometric Brownian motion using SVector representation.
"""
simulate_gbm_1d_svector(; kwargs...) = 
    simulate_gbm_svector(SVector{1,Float64}; kwargs...)

"""
    simulate_gbm_1d_array(; kwargs...)

Convenience function for 1D geometric Brownian motion using array representation.
"""
simulate_gbm_1d_array(; kwargs...) = 
    simulate_gbm_array(Float64, 1; kwargs...)