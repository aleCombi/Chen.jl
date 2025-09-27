using Revise,PathSignatures
using GLM, DataFrames
using Random, Statistics, StaticArrays

"""
Generate word names for dense tensor signature features using existing infrastructure
"""
function get_signature_feature_names(dim::Int, max_level::Int)
    feature_names = String[]
    
    # Generate words for each level
    for level in 1:max_level
        level_size = dim^level
        for pos in 1:level_size
            # Convert linear position to multi-index (base-d representation)
            indices = Int[]
            temp_pos = pos - 1  # 0-based
            for _ in 1:level
                pushfirst!(indices, (temp_pos % dim) + 1)  # 1-based indices
                temp_pos ÷= dim
            end
            
            # Create word string: e1, e2 for dim=2 become "e1", "e2", "e1e1", "e1e2", etc.
            word_str = join(["e$i" for i in indices])
            push!(feature_names, word_str)
        end
    end
    
    return feature_names
end

"""
Simple call option payoff regression against path signatures.
We regress (S_T - K) against the signature of time-augmented paths (t, S_t).
"""
function call_option_regression_experiment(;
    n_train = 600,
    n_test = 200,
    K = 1.0,           # strike price  
    horizon = 1.0,     # time to maturity
    n_steps = 50,      # path discretization
    signature_level = 3,
    seed = 42
)
    Random.seed!(seed)
    
    println("=== Call Option Payoff Regression ===")
    println("Strike K = $K, Maturity T = $horizon")
    println("Training: $n_train paths, Test: $n_test paths, Steps: $n_steps")
    println("Signature level: $signature_level")
    println()
    
    # Generate training data
    println("Generating training data...")
    S_0 = SVector(K)
    train_ensemble = simulate_brownian_svector(
        SVector{1,Float64}; 
        n_paths = n_train,
        horizon = horizon,
        n_steps = n_steps,
        x0 = S_0
    )
    
    # Generate test data  
    println("Generating test data...")
    test_ensemble = simulate_brownian_svector(
        SVector{1,Float64}; 
        n_paths = n_test,
        horizon = horizon,
        n_steps = n_steps,
        x0 = S_0
    )
    
    # Convert training data to time-augmented paths (t, S_t)
    train_time_augmented_paths = Vector{Vector{SVector{2,Float64}}}(undef, n_train)
    dt = horizon / n_steps
    
    @inbounds for i in 1:n_train
        path_1d = train_ensemble.paths[i]
        path_2d = Vector{SVector{2,Float64}}(undef, length(path_1d))
        
        for j in eachindex(path_1d)
            t = (j - 1) * dt  # time at step j
            S_t = path_1d[j][1]  # stock price at time t
            path_2d[j] = SVector(t, S_t)
        end
        
        train_time_augmented_paths[i] = path_2d
    end
    
    # Convert test data to time-augmented paths (t, S_t)
    test_time_augmented_paths = Vector{Vector{SVector{2,Float64}}}(undef, n_test)
    
    @inbounds for i in 1:n_test
        path_1d = test_ensemble.paths[i]
        path_2d = Vector{SVector{2,Float64}}(undef, length(path_1d))
        
        for j in eachindex(path_1d)
            t = (j - 1) * dt  # time at step j
            S_t = path_1d[j][1]  # stock price at time t
            path_2d[j] = SVector(t, S_t)
        end
        
        test_time_augmented_paths[i] = path_2d
    end
    
    # Create ensembles for time-augmented paths
    train_augmented_ensemble = SVectorEnsemble{2,Float64}(train_time_augmented_paths)
    test_augmented_ensemble = SVectorEnsemble{2,Float64}(test_time_augmented_paths)
    
    # Compute training payoffs: S_T - K
    train_payoffs = Vector{Float64}(undef, n_train)
    @inbounds for i in 1:n_train
        S_T = train_time_augmented_paths[i][end][2]  # final stock price
        train_payoffs[i] = S_T - K
    end
    
    # Compute test payoffs: S_T - K
    test_payoffs = Vector{Float64}(undef, n_test)
    @inbounds for i in 1:n_test
        S_T = test_time_augmented_paths[i][end][2]  # final stock price
        test_payoffs[i] = S_T - K
    end
    
    println("Training payoff statistics:")
    println("  Mean: $(round(mean(train_payoffs), digits=4))")
    println("  Std:  $(round(std(train_payoffs), digits=4))")
    println("  Positive%: $(round(100 * mean(train_payoffs .> 0), digits=1))%")
    
    println("Test payoff statistics:")
    println("  Mean: $(round(mean(test_payoffs), digits=4))")
    println("  Std:  $(round(std(test_payoffs), digits=4))")
    println("  Positive%: $(round(100 * mean(test_payoffs .> 0), digits=1))%")
    println()
    
    # Compute signatures
    println("Computing training signatures...")
    train_signatures = [Tensor{Float64}(2, signature_level) for _ in 1:n_train]
    PathSignatures.batch_signatures!(train_signatures, train_augmented_ensemble)
    
    println("Computing test signatures...")
    test_signatures = [Tensor{Float64}(2, signature_level) for _ in 1:n_test]
    PathSignatures.batch_signatures!(test_signatures, test_augmented_ensemble)
    
    # Extract signature features (skip level-0 constant term)
    function extract_features(sig::Tensor{Float64})
        # Just take all coefficients except the level-0 constant
        start_idx = sig.offsets[2] + 1  # skip level-0
        return sig.coeffs[start_idx:end]
    end
    
    # Build feature matrices  
    train_feature_matrix = reduce(hcat, [extract_features(sig) for sig in train_signatures])'
    test_feature_matrix = reduce(hcat, [extract_features(sig) for sig in test_signatures])'
    n_features = size(train_feature_matrix, 2)
    
    # Generate meaningful feature names using tensor words
    # For 2D (time, stock), e1 = time component, e2 = stock component
    feature_names = get_signature_feature_names(2, signature_level)
    
    println("Training feature matrix size: $(size(train_feature_matrix))")
    println("Test feature matrix size: $(size(test_feature_matrix))")
    println("Feature names: $feature_names")
    println()
    
    # Create DataFrame for GLM (training only)
    train_df = DataFrame(train_feature_matrix, feature_names)
    train_df.payoff = train_payoffs
    
    # Fit linear regression using GLM on training data
    println("Fitting linear regression on training data...")
    
    # Create formula dynamically
    feature_formula = join(feature_names, " + ")
    formula_str = "payoff ~ " * feature_formula
    formula = eval(Meta.parse("@formula($formula_str)"))
    
    model = lm(formula, train_df)
    
    # Print results
    println("\n" * "="^50)
    println(model)
    println("="^50)
    
    # Make predictions
    train_pred = predict(model)
    
    # For test predictions, create DataFrame
    test_df = DataFrame(test_feature_matrix, feature_names)
    test_pred = predict(model, test_df)
    
    # Compute training metrics
    train_residuals = train_payoffs - train_pred
    train_ss_res = sum(train_residuals.^2)
    train_ss_tot = sum((train_payoffs .- mean(train_payoffs)).^2)
    train_r2 = 1 - train_ss_res / train_ss_tot
    train_rmse = sqrt(mean(train_residuals.^2))
    train_mae = mean(abs.(train_residuals))
    
    # Compute test metrics
    test_residuals = test_payoffs - test_pred
    test_ss_res = sum(test_residuals.^2)
    test_ss_tot = sum((test_payoffs .- mean(test_payoffs)).^2)
    test_r2 = 1 - test_ss_res / test_ss_tot
    test_rmse = sqrt(mean(test_residuals.^2))
    test_mae = mean(abs.(test_residuals))
    
    println("\nTraining Metrics:")
    println("R² = $(round(train_r2, digits=4))")
    println("RMSE = $(round(train_rmse, digits=4))")
    println("MAE = $(round(train_mae, digits=4))")
    
    println("\nTest Metrics:")
    println("R² = $(round(test_r2, digits=4))")
    println("RMSE = $(round(test_rmse, digits=4))")
    println("MAE = $(round(test_mae, digits=4))")
    
    # Check generalization
    r2_gap = train_r2 - test_r2
    println("\nGeneralization:")
    println("R² gap (train - test) = $(round(r2_gap, digits=4))")
    if r2_gap > 0.1
        println("⚠️  Potential overfitting detected")
    else
        println("✅ Good generalization")
    end
    
    # Feature importance (absolute coefficients)
    coeffs = coef(model)[2:end]  # skip intercept
    importance = abs.(coeffs)
    perm = sortperm(importance, rev=true)
    
    println("\nTop Features (by |coefficient|):")
    for i in 1:min(5, length(coeffs))
        idx = perm[i]
        println("  $(feature_names[idx]): $(round(coeffs[idx], digits=4))")
    end
    
    println("\nInterpretation:")
    println("e1 = time component ∫dt")
    println("e2 = stock component ∫dS_t") 
    println("e1e2 = ∫∫dt⊗dS_t (time-stock interaction)")
    println("e2e2 = ∫∫dS_t⊗dS_t (quadratic variation)")
    
    return (
        model = model,
        train_payoffs = train_payoffs,
        test_payoffs = test_payoffs,
        train_predictions = train_pred,
        test_predictions = test_pred,
        train_features = train_feature_matrix,
        test_features = test_feature_matrix,
        feature_names = feature_names,
        train_r2 = train_r2,
        test_r2 = test_r2,
        train_rmse = train_rmse,
        test_rmse = test_rmse,
        train_mae = train_mae,
        test_mae = test_mae
    )
end

# Run the experiment
result = call_option_regression_experiment(
    n_train = 600,
    n_test = 200,
    signature_level = 3,
    K = 1.0
);