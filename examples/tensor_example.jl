using StochasticDiffEq, DiffEqFinancial, Chen, StaticArrays, LinearAlgebra

function augment_brownian(times::Vector{Float64}, W::Vector{Float64})
    @assert length(times) == length(W) "Time and Brownian vectors must have same length"
    
    return [SVector{2, Float64}(t, w) for (t, w) in zip(times, W)]
end

function simulate_ou_process(x0, κ, θ, η, T::Float64; dt=0.01, seed=123)
    tspan = (0.0, T)
    prob = OrnsteinUhlenbeckProblem(κ, θ, η, x0, tspan; seed=seed)
    
    sol = solve(prob, EM(), dt=dt, save_noise=true)
    return sol
end

x0, κ, θ, η = 1.0, 2.0, 0.5, 0.3
T = 1.0

# Simulate actual OU process
ou_sol = simulate_ou_process(x0, κ, θ, η, T, dt=0.01)
simulated_brownian = ou_sol.W
augmented_brownian = augment_brownian(ou_sol.W.t, ou_sol.W.u)
simulated_ou = ou_sol.u

max_order = 5
signature_vals = [vcat([1.0], signature_path(augmented_brownian[1:i], max_order)) for i in 2:length(augmented_brownian)]
tensor_ou = [ornstein_uhlenbeck_time_dependent(x0, κ, θ, η, t, max_order=max_order) for t in ou_sol.W.t[2:end]]
flat_tensor_ou = [tensor_to_vector(tensor, standard_word_map(max_order, 2)) for tensor in tensor_ou]

# Compute bracket for each time step
bracket_results = [dot(sig_vec, flat_tensor_vec) for (sig_vec, flat_tensor_vec) in zip(signature_vals, flat_tensor_ou)]