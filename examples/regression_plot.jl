using Revise,PathSignatures
using GLM, DataFrames, GLMNet
using Random, Statistics, StaticArrays, Plots

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
Check if a word contains only time component (e1) - should be filtered out
For time-augmented paths (t, S_t), e1 is time, e2 is price
"""
function is_time_only_word(word_str::String)
    # A word is time-only if it contains only "e1" (like "e1", "e1e1", "e1e1e1", etc.)
    # We check if all characters after removing "e" and "1" are gone
    cleaned = replace(word_str, "e1" => "")
    return isempty(cleaned)
end

"""
Simple call option payoff regression against path signatures.
We regress (S_T - K) against the signature of time-augmented paths (t, S_t).

Supports multiple regression methods:
- :ols - Ordinary Least Squares (default)
- :ridge - Ridge regression (L2 penalty)
- :lasso - Lasso regression (L1 penalty)
- :elasticnet - Elastic Net (L1 + L2 penalty)
"""
function call_option_regression_experiment(;
    n_train = 600,
    n_test = 200,
    S0 = 1.0,           #spot price
    K = 1.0,           # strike price  
    horizon = 1.0,     # time to maturity
    n_steps = 50,      # path discretization
    μ = 0.05,          # GBM drift parameter
    σ = 0.2,           # GBM volatility parameter
    signature_level = 5,
    underlying_func = path -> path[end][2], #default: euro payoff
    payoff_func = (path, strike) -> max.(underlying_func(path) - strike, 0),  # default: call payoff
    verbose = false,
    regression_method = :ols,  # :ols, :ridge, :lasso, :elasticnet
    alpha = 1.0,       # regularization strength (for ridge/lasso/elasticnet)
    lambda = nothing,  # if provided, use this lambda instead of CV
    l1_ratio = 0.5     # for elasticnet: 0=ridge, 1=lasso
)
    # No Random.seed! - let it be random each time
    logS_0 = SVector(log(S0))
    # Generate training data
    train_ensemble = PathSignatures.simulate_loggbm_svector(
        SVector{1,Float64}; 
        n_paths = n_train,
        horizon = horizon,
        n_steps = n_steps,
        y0 = logS_0,
        μ = SVector(μ),
        σ = SVector(σ)
    )
    train_ensemble = exp(train_ensemble)

    # Generate test data  
    test_ensemble = PathSignatures.simulate_loggbm_svector(
        SVector{1,Float64}; 
        n_paths = n_test,
        horizon = horizon,
        n_steps = n_steps,
        y0 = logS_0,
        μ = SVector(μ),
        σ = SVector(σ)
    ) 
    test_ensemble = exp(test_ensemble)
    
    # Convert to time-augmented paths (t, S_t)
    dt = horizon / n_steps
    
    function augment_path(path_1d)
        path_2d = Vector{SVector{2,Float64}}(undef, length(path_1d))
        @inbounds for j in eachindex(path_1d)
            t = (j - 1) * dt
            S_t = path_1d[j][1]
            path_2d[j] = SVector(t, S_t)
        end
        return path_2d
    end
    
    train_time_augmented_paths = [augment_path(p) for p in train_ensemble.paths]
    test_time_augmented_paths = [augment_path(p) for p in test_ensemble.paths]
    
    # Create ensembles
    train_augmented_ensemble = SVectorEnsemble{2,Float64}(train_time_augmented_paths)
    test_augmented_ensemble = SVectorEnsemble{2,Float64}(test_time_augmented_paths)
    
    # Compute payoffs
    train_payoffs = [payoff_func(p, K) for p in train_time_augmented_paths]
    test_payoffs = [payoff_func(p, K) for p in test_time_augmented_paths]

    # Compute signatures
    train_signatures = [Tensor{Float64}(2, signature_level) for _ in 1:n_train]
    PathSignatures.batch_signatures!(train_signatures, train_augmented_ensemble)
    
    test_signatures = [Tensor{Float64}(2, signature_level) for _ in 1:n_test]
    PathSignatures.batch_signatures!(test_signatures, test_augmented_ensemble)
    
    # Extract features (skip level-0 constant term)
    function extract_features(sig::Tensor{Float64})
        start_idx = sig.offsets[2] + 1
        return sig.coeffs[start_idx:end]
    end
    
    # Build feature matrices  
    train_feature_matrix = reduce(hcat, [extract_features(sig) for sig in train_signatures])'
    test_feature_matrix = reduce(hcat, [extract_features(sig) for sig in test_signatures])'
    
    # Generate feature names
    all_feature_names = get_signature_feature_names(2, signature_level)
    
    # Filter out time-only features (e.g., e1, e1e1, e1e1e1, etc.)
    # Keep only features that involve the price process (e2)
    feature_mask = .!is_time_only_word.(all_feature_names)
    filtered_feature_names = all_feature_names[feature_mask]
    
    # Apply filter to feature matrices
    train_feature_matrix = train_feature_matrix[:, feature_mask]
    test_feature_matrix = test_feature_matrix[:, feature_mask]
    
    n_total_features = length(all_feature_names)
    n_filtered_features = length(filtered_feature_names)
    n_removed = n_total_features - n_filtered_features
    
    # Create DataFrame for GLM
    train_df = DataFrame(train_feature_matrix, filtered_feature_names)
    train_df.payoff = train_payoffs
    
    # Fit regression model based on method
    if regression_method == :ols
        # Standard OLS using GLM
        feature_formula = join(filtered_feature_names, " + ")
        formula_str = "payoff ~ " * feature_formula
        formula = eval(Meta.parse("@formula($formula_str)"))
        
        model = lm(formula, train_df)
        train_pred = GLM.predict(model)
        
        test_df = DataFrame(test_feature_matrix, filtered_feature_names)
        test_pred = GLM.predict(model, test_df)
        
    elseif regression_method in [:ridge, :lasso, :elasticnet]
        # Use GLMNet for regularized regression
        X_train = train_feature_matrix
        y_train = train_payoffs
        X_test = test_feature_matrix
        
        # Set alpha parameter for GLMNet
        # alpha=0 is ridge, alpha=1 is lasso, 0<alpha<1 is elasticnet
        glmnet_alpha = if regression_method == :ridge
            0.0
        elseif regression_method == :lasso
            1.0
        else  # elasticnet
            l1_ratio
        end
        
        if lambda === nothing
            # Use cross-validation to select lambda
            cv_model = glmnetcv(X_train, y_train; alpha=glmnet_alpha)
            model = cv_model
            
            # Get predictions using lambda.min (minimum CV error)
            train_pred = GLMNet.predict(cv_model, X_train)
            test_pred = GLMNet.predict(cv_model, X_test)
        else
            # Use specified lambda
            model = glmnet(X_train, y_train; alpha=glmnet_alpha)
            train_pred = GLMNet.predict(model, X_train; s=[lambda])
            test_pred = GLMNet.predict(model, X_test; s=[lambda])
        end
        
    else
        error("Unknown regression method: $regression_method. Use :ols, :ridge, :lasso, or :elasticnet")
    end
    
    # Compute metrics
    train_residuals = train_payoffs - train_pred
    train_r2 = 1 - sum(train_residuals.^2) / sum((train_payoffs .- mean(train_payoffs)).^2)
    train_rmse = sqrt(mean(train_residuals.^2))
    train_mae = mean(abs.(train_residuals))
    
    test_residuals = test_payoffs - test_pred
    test_r2 = 1 - sum(test_residuals.^2) / sum((test_payoffs .- mean(test_payoffs)).^2)
    test_rmse = sqrt(mean(test_residuals.^2))
    test_mae = mean(abs.(test_residuals))
    
    # Print clean results
    println("\n" * "="^60)
    println("CALL OPTION PAYOFF REGRESSION ($(uppercase(string(regression_method))))")
    println("="^60)
    println("Setup: K=$K, T=$horizon, Steps=$n_steps, Signature Level=$signature_level")
    println("GBM:   μ=$μ, σ=$σ")
    println("Data:  Train=$n_train paths, Test=$n_test paths")
    println("Features: $n_filtered_features used ($n_removed time-only features removed)")
    if regression_method != :ols
        if lambda === nothing
            println("Regularization: CV-selected λ")
        else
            println("Regularization: λ=$lambda")
        end
    end
    println("-"^60)
    
    println("\nPayoff Statistics:")
    println("  Train: Mean=$(round(mean(train_payoffs), digits=4)), Std=$(round(std(train_payoffs), digits=4))")
    println("  Test:  Mean=$(round(mean(test_payoffs), digits=4)), Std=$(round(std(test_payoffs), digits=4))")
    
    println("\nRegression Performance:")
    println("  Training:   R²=$(round(train_r2, digits=4))  RMSE=$(round(train_rmse, digits=4))  MAE=$(round(train_mae, digits=4))")
    println("  Test:       R²=$(round(test_r2, digits=4))  RMSE=$(round(test_rmse, digits=4))  MAE=$(round(test_mae, digits=4))")
    
    r2_gap = train_r2 - test_r2
    generalization_status = r2_gap > 0.1 ? "⚠️  Overfitting" : "✅ Good"
    println("  R² Gap:     $(round(r2_gap, digits=4)) ($generalization_status)")
    
    # Top features
    if regression_method == :ols
        coeffs = GLM.coef(model)[2:end]  # skip intercept
    else
        # For GLMNet, extract coefficients at the selected lambda
        if lambda === nothing
            # Use lambda.min from CV
            coeffs = vec(model.path.betas[:, argmin(model.meanloss)])
        else
            # Use specified lambda - find closest one
            lambda_idx = argmin(abs.(model.lambda .- lambda))
            coeffs = vec(model.betas[:, lambda_idx])
        end
    end
    
    importance = abs.(coeffs)
    perm = sortperm(importance, rev=true)
    
    # Count non-zero coefficients (for regularized methods)
    n_nonzero = sum(coeffs .!= 0)
    
    println("\nTop 5 Features (by |coefficient|):")
    for i in 1:min(5, length(coeffs))
        idx = perm[i]
        println("  $(rpad(filtered_feature_names[idx], 10)) → $(round(coeffs[idx], digits=6))")
    end
    
    if regression_method in [:ridge, :lasso, :elasticnet]
        println("\nSparsity: $n_nonzero / $(length(coeffs)) non-zero coefficients")
    end
    
    println("\nFeature Interpretation:")
    println("  e1     = time ∫dt (REMOVED)")
    println("  e2     = stock ∫dS_t") 
    println("  e1e2   = time-stock ∫∫dt⊗dS_t")
    println("  e2e2   = quadratic variation ∫∫dS_t⊗dS_t")
    println("  All pure time terms (e1, e1e1, ...) are excluded")
    println("="^60 * "\n")
    
    if verbose
        if regression_method == :ols
            println("\nFull Model Summary:")
            println(model)
        elseif regression_method in [:ridge, :lasso, :elasticnet]
            println("\nRegularization Path Info:")
            println("  Number of lambdas: $(length(model.lambda))")
            if lambda === nothing
                println("  Best lambda (CV): $(model.lambda[argmin(model.meanloss)])")
                println("  Min CV error: $(minimum(model.meanloss))")
            end
        end
    end
    
    return (
    model = model,
    train_payoffs = train_payoffs,
    test_payoffs = test_payoffs,
    train_predictions = train_pred,
    test_predictions = test_pred,
    test_underlying_values = [underlying_func(p) for p in test_time_augmented_paths],  # ADD THIS LINE
    train_features = train_feature_matrix,
    test_features = test_feature_matrix,
    feature_names = filtered_feature_names,
    all_feature_names = all_feature_names,
    n_features_removed = n_removed,
    coefficients = coeffs,
    train_r2 = train_r2,
    test_r2 = test_r2,
    train_rmse = train_rmse,
    test_rmse = test_rmse,
    train_mae = train_mae,
    test_mae = test_mae
)
end

# Run experiments with different methods
println("\n" * "="^60)
println("COMPARING REGRESSION METHODS")
println("="^60)

asian_arith_call = function (path, K)
    # path is Vector{SVector{2,Float64}} with entries (t, S_t)
    avg = mean(p[2] for p in path)
    max(avg - K, 0.0)
end

underlying_func(my_path) = mean(p[2] for p in my_path) # euro option
payoff_func(path, strike) = max.(underlying_func(path) - strike, 0)  # call option
# Common parameters for all experiments
# Common parameters for all experiments
common_params = (
    n_train = 10000,
    n_test = 1000,
    signature_level = 8,
    K = 1.0,
    horizon = 1.0,        # ADD THIS
    n_steps = 100,
    verbose = false,
    underlying_func = underlying_func, # euro option
    payoff_func = payoff_func,  # call option
    μ = 0.0,
    σ = 0.2
)

# OLS
result_ols = call_option_regression_experiment(;
    common_params...,
    regression_method = :ols
);

scatter(result_ols.test_underlying_values, result_ols.test_predictions, 
        markersize=2, label="Predictions", alpha=0.6)
scatter!(result_ols.test_underlying_values, result_ols.test_payoffs, 
        markersize=2, label="True Payoffs", alpha=0.6)