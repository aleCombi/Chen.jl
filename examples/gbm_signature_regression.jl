# Geometric Brownian Motion Signature Regression Example
#
# Demonstrates:
# 1. Simulating N paths of geometric Brownian motion
# 2. Time augmentation and batch signature computation
# 3. Regressing against call option payoff (S_T - K)^+
# 4. Visualization of regression results

using ChenSignatures
using GLM
using Plots
using Random
using Statistics
using LinearAlgebra

# ============================================================================
# Geometric Brownian Motion Simulation
# ============================================================================

# Simulate a single GBM path: dS = μS dt + σS dW
function simulate_gbm_path(S₀::Real, μ::Real, σ::Real, T::Real, N::Int;
                           rng=Random.GLOBAL_RNG)
    dt = T / (N - 1)
    path = Vector{Float64}(undef, N)
    path[1] = S₀

    for i in 2:N
        dW = sqrt(dt) * randn(rng)
        path[i] = path[i-1] * exp((μ - 0.5 * σ^2) * dt + σ * dW)
    end

    return path
end

# Simulate multiple GBM paths in batch format (N × 1 × num_paths)
function simulate_gbm_paths(S₀::Real, μ::Real, σ::Real, T::Real, N::Int,
                           num_paths::Int; rng=Random.GLOBAL_RNG)
    paths = Array{Float64, 3}(undef, N, 1, num_paths)

    for i in 1:num_paths
        paths[:, 1, i] = simulate_gbm_path(S₀, μ, σ, T, N; rng=rng)
    end

    return paths
end

# Time-augment paths: add time as explicit coordinate (1D → 2D)
# This enriches the signature feature space significantly
function time_augment_paths(paths::Array{Float64, 3}, T::Real)
    N, d, num_paths = size(paths)
    @assert d == 1 "Expected 1D paths, got dimension $d"

    augmented = Array{Float64, 3}(undef, N, 2, num_paths)
    time_grid = range(0, T, length=N)

    for i in 1:num_paths
        augmented[:, 1, i] = time_grid      # Time coordinate
        augmented[:, 2, i] = paths[:, 1, i] # Stock price
    end

    return augmented
end

# ============================================================================
# Payoff Functions
# ============================================================================

# Call option payoff: (S_T - K)^+
call_payoff(S_T::Real, K::Real) = max(S_T - K, 0.0)

# Extract terminal stock prices from batch of paths
function extract_terminal_values(paths::Array{Float64, 3})
    N, _, num_paths = size(paths)
    return [paths[N, 1, i] for i in 1:num_paths]
end

# Compute call option payoffs for all paths
function compute_payoffs(paths::Array{Float64, 3}, K::Real)
    terminal_values = extract_terminal_values(paths)
    return call_payoff.(terminal_values, K)
end

# ============================================================================
# Signature Computation
# ============================================================================

# Compute signatures for batch of paths using efficient batch processing
function compute_signatures_batch(paths::Array{Float64, 3}, level::Int)
    return sig(paths, level; threaded=true)
end

# ============================================================================
# Regression
# ============================================================================

# Fit linear regression: signatures → payoffs
function fit_signature_regression(signatures::Matrix{Float64}, targets::Vector{Float64})
    X = signatures'  # GLM expects (num_samples, num_features)
    return lm(X, targets)
end

# Predict using fitted model
function predict_regression(model, signatures::Matrix{Float64})
    X = signatures'
    return predict(model, X)
end

# Compute R² score
function compute_r_squared(y_true::Vector{Float64}, y_pred::Vector{Float64})
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - ss_res / ss_tot
end

# ============================================================================
# Visualization
# ============================================================================

# Plot predicted vs actual payoffs as function of terminal stock price
function plot_regression_results(S_T::Vector{Float64},
                                payoffs_true::Vector{Float64},
                                payoffs_pred::Vector{Float64},
                                K::Real)
    # Sort by terminal price for cleaner visualization
    perm = sortperm(S_T)
    S_T_sorted = S_T[perm]
    true_sorted = payoffs_true[perm]
    pred_sorted = payoffs_pred[perm]

    # Create plot
    plt = scatter(S_T_sorted, true_sorted,
                  label="True Payoff",
                  alpha=0.6,
                  xlabel="Terminal Stock Price (S_T)",
                  ylabel="Payoff",
                  title="Signature Regression: Predicted vs True Payoffs",
                  legend=:topleft,
                  markersize=4)

    scatter!(plt, S_T_sorted, pred_sorted,
             label="Predicted Payoff",
             alpha=0.6,
             markersize=4)

    # Add strike price line
    vline!(plt, [K],
           label="Strike (K=$K)",
           linestyle=:dash,
           linewidth=2,
           color=:black)

    return plt
end

# ============================================================================
# Main Example
# ============================================================================

function run_gbm_signature_regression_example(;
    S₀=100.0,
    μ=0.05,
    σ=0.2,
    T=1.0,
    K=100.0,
    N=50,
    train_paths=1000,
    test_paths=200,
    plot_paths=500,  # Additional paths for plotting
    sig_level=4,
    seed=42
)
    println("=" ^ 70)
    println("Geometric Brownian Motion Signature Regression")
    println("=" ^ 70)
    println()

    rng = MersenneTwister(seed)

    # ========================================================================
    # 1. Simulate training paths
    # ========================================================================
    println("1. Simulating $train_paths training paths...")
    train_paths_data = simulate_gbm_paths(S₀, μ, σ, T, N, train_paths; rng=rng)
    println("   Shape: $(size(train_paths_data))")
    println()

    # ========================================================================
    # 2. Time-augment and compute signatures
    # ========================================================================
    println("2. Time-augmenting paths and computing signatures (level $sig_level)...")
    train_paths_augmented = time_augment_paths(train_paths_data, T)
    train_signatures = compute_signatures_batch(train_paths_augmented, sig_level)
    sig_dim = size(train_signatures, 1)
    println("   Signature dimension: $sig_dim")
    println()

    # ========================================================================
    # 3. Compute payoffs and fit regression
    # ========================================================================
    println("3. Fitting regression model (signatures → payoffs)...")
    train_payoffs = compute_payoffs(train_paths_data, K)
    model = fit_signature_regression(train_signatures, train_payoffs)
    train_predictions = predict_regression(model, train_signatures)
    train_r2 = compute_r_squared(train_payoffs, train_predictions)
    println("   Training R²: $(round(train_r2, digits=4))")
    println()

    # ========================================================================
    # 4. Test on separate test set
    # ========================================================================
    println("4. Testing on $test_paths new paths...")
    test_paths_data = simulate_gbm_paths(S₀, μ, σ, T, N, test_paths; rng=rng)
    test_paths_augmented = time_augment_paths(test_paths_data, T)
    test_signatures = compute_signatures_batch(test_paths_augmented, sig_level)
    test_payoffs = compute_payoffs(test_paths_data, K)
    test_predictions = predict_regression(model, test_signatures)
    test_r2 = compute_r_squared(test_payoffs, test_predictions)
    println("   Test R²: $(round(test_r2, digits=4))")
    println("   MAE: $(round(mean(abs.(test_payoffs .- test_predictions)), digits=2))")
    println()

    # ========================================================================
    # 5. Generate visualization on separate plot set
    # ========================================================================
    println("5. Generating visualization on $plot_paths new paths...")
    plot_paths_data = simulate_gbm_paths(S₀, μ, σ, T, N, plot_paths; rng=rng)
    plot_paths_augmented = time_augment_paths(plot_paths_data, T)
    plot_signatures = compute_signatures_batch(plot_paths_augmented, sig_level)
    plot_payoffs = compute_payoffs(plot_paths_data, K)
    plot_predictions = predict_regression(model, plot_signatures)
    plot_S_T = extract_terminal_values(plot_paths_data)

    plt = plot_regression_results(plot_S_T, plot_payoffs, plot_predictions, K)
    display(plt)
    savefig(plt, "gbm_signature_regression.png")
    println("   Plot saved as 'gbm_signature_regression.png'")
    println()

    # ========================================================================
    # Summary
    # ========================================================================
    println("=" ^ 70)
    println("Summary")
    println("=" ^ 70)
    println("GBM Parameters: S₀=$S₀, μ=$μ, σ=$σ, T=$T")
    println("Option: Call with strike K=$K")
    println("Paths: $train_paths training, $test_paths test, $plot_paths plot")
    println("Time augmentation: 1D → 2D (time, price)")
    println("Signature level: $sig_level (dimension: $sig_dim)")
    println()
    println("Results:")
    println("  Training R²: $(round(train_r2, digits=4))")
    println("  Test R²:     $(round(test_r2, digits=4))")
    println("=" ^ 70)

    return (model=model, train_r2=train_r2, test_r2=test_r2, plot=plt)
end

# ============================================================================
# Run the example
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    run_gbm_signature_regression_example()
end
