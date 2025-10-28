using Revise, PathSignatures
using GLM, DataFrames, GLMNet
using Random, Statistics, StaticArrays, Plots
using Colors

# ------------------------------------------------------------
# Feature naming helpers
# ------------------------------------------------------------

"""
    get_signature_feature_names(dim::Int, max_level::Int) -> Vector{String}

Generate word names for dense tensor signature features.
For dim=2, returns: e1, e2, e1e1, e1e2, e2e1, e2e2, ...
"""
function get_signature_feature_names(dim::Int, max_level::Int)
    feature_names = String[]
    for level in 1:max_level
        level_size = dim^level
        for pos in 1:level_size
            # Convert linear index (1-based) to base-d digits (1..dim)
            digits = Int[]
            tmp = pos - 1
            for _ in 1:level
                pushfirst!(digits, (tmp % dim) + 1)
                tmp ÷= dim
            end
            push!(feature_names, join(("e$k" for k in digits)))
        end
    end
    return feature_names
end

"""
    is_time_only_word(word::String) -> Bool

For time-augmented paths with channels (t, S_t) mapped to (e1, e2),
returns true if the word contains only e1 (e.g., "e1", "e1e1", ...).
"""
function is_time_only_word(word_str::String)
    cleaned = replace(word_str, "e1" => "")
    return isempty(cleaned)
end

# ------------------------------------------------------------
# Core experiment
# ------------------------------------------------------------

"""
    call_option_regression_experiment(; kwargs...) -> NamedTuple

Simulate GBM paths for a single underlying; build time-augmented paths (t, S_t);
compute signatures up to `signature_level`; filter out time-only features; and
regress a payoff against the remaining features.

Supported regression methods:
- :ols         Ordinary Least Squares (GLM)
- :ridge       Ridge (GLMNet; alpha=0)
- :lasso       Lasso (GLMNet; alpha=1)
- :elasticnet  Elastic net (GLMNet; alpha=l1_ratio)
"""
function call_option_regression_experiment(;
    n_train::Int = 600,
    n_test::Int = 200,
    S0::Float64 = 1.0,
    K::Float64 = 1.0,
    horizon::Float64 = 1.0,
    n_steps::Int = 50,
    μ::Float64 = 0.05,
    σ::Float64 = 0.2,
    signature_level::Int = 5,
    underlying_func::Function = path -> path[end][2], # Euro payoff: terminal S
    payoff_func::Function = (path, strike) -> max(underlying_func(path) - strike, 0.0),
    regression_method::Symbol = :ols,
    alpha::Float64 = 1.0,
    lambda = nothing,
    l1_ratio::Float64 = 0.5,
    verbose::Bool = false,
    # ---- NEW ----
    normalize_features::Bool = true,
    normalize_target::Bool = false,
    path_transform::Function = identity,
)
    # ---- Simulate GBM in log-space, then exponentiate ----
    logS0 = SVector(log(S0))
    train_ens_log = PathSignatures.simulate_loggbm_svector(
        SVector{1,Float64};
        n_paths = n_train, horizon = horizon, n_steps = n_steps,
        y0 = logS0, μ = SVector(μ), σ = SVector(σ),
    )
    test_ens_log = PathSignatures.simulate_loggbm_svector(
        SVector{1,Float64};
        n_paths = n_test, horizon = horizon, n_steps = n_steps,
        y0 = logS0, μ = SVector(μ), σ = SVector(σ),
    )
    train_ens = path_transform(train_ens_log)
    test_ens  = path_transform(test_ens_log)

    # ---- Time augmentation: (t, S_t) ----
    dt = horizon / n_steps
    augment = function(path_1d)
        out = Vector{SVector{2,Float64}}(undef, length(path_1d))
        @inbounds for j in eachindex(path_1d)
            t = (j - 1) * dt
            S_t = path_1d[j][1]
            out[j] = SVector(t, S_t)
        end
        out
    end
    train_paths_2d = [augment(p) for p in train_ens.paths]
    test_paths_2d  = [augment(p) for p in test_ens.paths]

    train_ens_2d = SVectorEnsemble{2,Float64}(train_paths_2d)
    test_ens_2d  = SVectorEnsemble{2,Float64}(test_paths_2d)

    # ---- Targets ----
    train_payoffs = [payoff_func(p, K) for p in train_paths_2d]
    test_payoffs  = [payoff_func(p, K) for p in test_paths_2d]

    # ---- Signatures ----
    train_sigs = [Tensor{Float64}(2, signature_level) for _ in 1:n_train]
    test_sigs  = [Tensor{Float64}(2, signature_level) for _ in 1:n_test]
    PathSignatures.batch_signatures!(train_sigs, train_ens_2d)
    PathSignatures.batch_signatures!(test_sigs,  test_ens_2d)

    # ---- Extract features (skip level-0 constant) ----
    extract_features = function(sig::Tensor{Float64})
        start_idx = sig.offsets[2] + 1
        @view sig.coeffs[start_idx:end]
    end
    train_X_full = reduce(hcat, (extract_features(sig) for sig in train_sigs))'
    test_X_full  = reduce(hcat, (extract_features(sig) for sig in test_sigs))'

    # ---- Name & filter features (remove pure-time) ----
    all_feature_names = get_signature_feature_names(2, signature_level)
    keep_mask = .!is_time_only_word.(all_feature_names)
    kept_names = all_feature_names[keep_mask]
    train_X = train_X_full[:, keep_mask]
    test_X  = test_X_full[:, keep_mask]

    # =========================
    # NORMALIZATION (NEW)
    # =========================
    # Feature standardization
    if normalize_features
        X_mean = mean(train_X; dims=1)
        X_std  = std(train_X;  dims=1)
        X_std[X_std .== 0] .= 1.0               # avoid division by zero
        train_X = (train_X .- X_mean) ./ X_std
        test_X  = (test_X  .- X_mean) ./ X_std  # use TRAIN stats
    end

    # Target standardization (optional; predictions are de-normalized later)
    y_mean = 0.0
    y_std  = 1.0
    if normalize_target
        y_mean = mean(train_payoffs)
        y_std  = std(train_payoffs)
        y_std == 0 && (y_std = 1.0)
        train_payoffs = (train_payoffs .- y_mean) ./ y_std
        # NOTE: do NOT normalize test_payoffs; keep ground-truth in original units
    end

    # ---- Fit model ----
    model, yhat_train, yhat_test = begin
        if regression_method == :ols
            df_train = DataFrame(train_X, kept_names)
            df_train.payoff = train_payoffs
            rhs = join(kept_names, " + ")
            f = eval(Meta.parse("@formula(payoff ~ $rhs)"))
            mdl = lm(f, df_train)
            df_test = DataFrame(test_X, kept_names)
            (mdl, GLM.predict(mdl), GLM.predict(mdl, df_test))

        elseif regression_method in (:ridge, :lasso, :elasticnet)
            α_glmnet = regression_method == :ridge ? 0.0 :
                       regression_method == :lasso ? 1.0 : l1_ratio
            if lambda === nothing
                cv = glmnetcv(train_X, train_payoffs; alpha=α_glmnet)
                (cv, GLMNet.predict(cv, train_X), GLMNet.predict(cv, test_X))
            else
                path = glmnet(train_X, train_payoffs; alpha=α_glmnet)
                (path,
                 GLMNet.predict(path, train_X; s=[lambda]),
                 GLMNet.predict(path, test_X; s=[lambda]))
            end
        else
            error("Unknown regression method: $regression_method")
        end
    end

    # De-normalize predictions if target was normalized
    if normalize_target
        yhat_train = yhat_train .* y_std .+ y_mean
        yhat_test  = yhat_test  .* y_std .+ y_mean
    end

    # ---- Metrics ----
    resid_train = train_payoffs .* (normalize_target ? y_std : 1) .+ (normalize_target ? y_mean : 0)
    resid_train .-= yhat_train
    resid_test  = test_payoffs .- yhat_test

    r2_train = 1 - sum(resid_train.^2) / sum(((train_payoffs .* (normalize_target ? y_std : 1) .+ (normalize_target ? y_mean : 0)) .- mean(train_payoffs .* (normalize_target ? y_std : 1) .+ (normalize_target ? y_mean : 0))).^2)
    r2_test  = 1 - sum(resid_test.^2)  / sum((test_payoffs .- mean(test_payoffs)).^2)
    rmse_train = sqrt(mean(resid_train.^2))
    rmse_test  = sqrt(mean(resid_test.^2))
    mae_train  = mean(abs.(resid_train))
    mae_test   = mean(abs.(resid_test))

    # ---- Coefficients (for reference) ----
    coeffs = if regression_method == :ols
        GLM.coef(model)[2:end]               # drop intercept
    else
        if lambda === nothing
            model.path.betas[:, argmin(model.meanloss)]
        else
            idx = argmin(abs.(model.lambda .- lambda))
            model.betas[:, idx]
        end
    end

    return (
        model = model,
        train_payoffs = normalize_target ? (train_payoffs .* y_std .+ y_mean) : train_payoffs,  # return in original units
        test_payoffs = test_payoffs,
        train_predictions = yhat_train,
        test_predictions = yhat_test,
        test_underlying_values = [underlying_func(p) for p in test_paths_2d],
        train_features = train_X,
        test_features = test_X,
        feature_names = kept_names,
        all_feature_names = all_feature_names,
        n_features_removed = length(all_feature_names) - length(kept_names),
        coefficients = coeffs,
        train_r2 = r2_train,
        test_r2 = r2_test,
        train_rmse = rmse_train,
        test_rmse = rmse_test,
        train_mae = mae_train,
        test_mae = mae_test,
    )
end
function call_option_regression_experiment(;
    n_train::Int = 600,
    n_test::Int = 200,
    S0::Float64 = 1.0,
    K::Float64 = 1.0,
    horizon::Float64 = 1.0,
    n_steps::Int = 50,
    μ::Float64 = 0.05,
    σ::Float64 = 0.2,
    signature_level::Int = 5,
    path_transform::Function = identity,
    underlying_func::Function,
    payoff_func::Function,
    regression_method::Symbol = :ols,
    alpha::Float64 = 1.0,
    lambda = nothing,
    l1_ratio::Float64 = 0.5,
    verbose::Bool = false,
    # ---- NEW ----
    normalize_features::Bool = true,
    normalize_target::Bool = false,
)
    # ---- Simulate GBM in log-space, then exponentiate ----
    logS0 = SVector(log(S0))
    train_ens_log = PathSignatures.simulate_loggbm_svector(
        SVector{1,Float64};
        n_paths = n_train, horizon = horizon, n_steps = n_steps,
        y0 = logS0, μ = SVector(μ), σ = SVector(σ),
    )
    test_ens_log = PathSignatures.simulate_loggbm_svector(
        SVector{1,Float64};
        n_paths = n_test, horizon = horizon, n_steps = n_steps,
        y0 = logS0, μ = SVector(μ), σ = SVector(σ),
    )
    train_ens = path_transform(train_ens_log)
    test_ens  = path_transform(test_ens_log)

    # ---- Time augmentation: (t, S_t) ----
    dt = horizon / n_steps
    augment = function(path_1d)
        out = Vector{SVector{2,Float64}}(undef, length(path_1d))
        @inbounds for j in eachindex(path_1d)
            t = (j - 1) * dt
            S_t = path_1d[j][1]
            out[j] = SVector(t, S_t)
        end
        out
    end
    train_paths_2d = [augment(p) for p in train_ens.paths]
    test_paths_2d  = [augment(p) for p in test_ens.paths]

    train_ens_2d = SVectorEnsemble{2,Float64}(train_paths_2d)
    test_ens_2d  = SVectorEnsemble{2,Float64}(test_paths_2d)

    # ---- Targets ----
    train_payoffs = [payoff_func(p, K) for p in train_paths_2d]
    test_payoffs  = [payoff_func(p, K) for p in test_paths_2d]

    # ---- Signatures ----
    train_sigs = [Tensor{Float64}(2, signature_level) for _ in 1:n_train]
    test_sigs  = [Tensor{Float64}(2, signature_level) for _ in 1:n_test]
    PathSignatures.batch_signatures!(train_sigs, train_ens_2d)
    PathSignatures.batch_signatures!(test_sigs,  test_ens_2d)

    # ---- Extract features (skip level-0 constant) ----
    extract_features = function(sig::Tensor{Float64})
        start_idx = sig.offsets[2] + 1
        @view sig.coeffs[start_idx:end]
    end
    train_X_full = reduce(hcat, (extract_features(sig) for sig in train_sigs))'
    test_X_full  = reduce(hcat, (extract_features(sig) for sig in test_sigs))'

    # ---- Name & filter features (remove pure-time) ----
    all_feature_names = get_signature_feature_names(2, signature_level)
    keep_mask = .!is_time_only_word.(all_feature_names)
    kept_names = all_feature_names[keep_mask]
    train_X = train_X_full[:, keep_mask]
    test_X  = test_X_full[:, keep_mask]

    # =========================
    # NORMALIZATION (NEW)
    # =========================
    # Feature standardization
    if normalize_features
        X_mean = mean(train_X; dims=1)
        X_std  = std(train_X;  dims=1)
        X_std[X_std .== 0] .= 1.0               # avoid division by zero
        train_X = (train_X .- X_mean) ./ X_std
        test_X  = (test_X  .- X_mean) ./ X_std  # use TRAIN stats
    end

    # Target standardization (optional; predictions are de-normalized later)
    y_mean = 0.0
    y_std  = 1.0
    if normalize_target
        y_mean = mean(train_payoffs)
        y_std  = std(train_payoffs)
        y_std == 0 && (y_std = 1.0)
        train_payoffs = (train_payoffs .- y_mean) ./ y_std
        # NOTE: do NOT normalize test_payoffs; keep ground-truth in original units
    end

    # ---- Fit model ----
    model, yhat_train, yhat_test = begin
        if regression_method == :ols
            df_train = DataFrame(train_X, kept_names)
            df_train.payoff = train_payoffs
            rhs = join(kept_names, " + ")
            f = eval(Meta.parse("@formula(payoff ~ $rhs)"))
            mdl = lm(f, df_train)
            df_test = DataFrame(test_X, kept_names)
            (mdl, GLM.predict(mdl), GLM.predict(mdl, df_test))

        elseif regression_method in (:ridge, :lasso, :elasticnet)
            α_glmnet = regression_method == :ridge ? 0.0 :
                       regression_method == :lasso ? 1.0 : l1_ratio
            if lambda === nothing
                cv = glmnetcv(train_X, train_payoffs; alpha=α_glmnet)
                (cv, GLMNet.predict(cv, train_X), GLMNet.predict(cv, test_X))
            else
                path = glmnet(train_X, train_payoffs; alpha=α_glmnet)
                (path,
                 GLMNet.predict(path, train_X; s=[lambda]),
                 GLMNet.predict(path, test_X; s=[lambda]))
            end
        else
            error("Unknown regression method: $regression_method")
        end
    end

    # De-normalize predictions if target was normalized
    if normalize_target
        yhat_train = yhat_train .* y_std .+ y_mean
        yhat_test  = yhat_test  .* y_std .+ y_mean
    end

    # ---- Metrics ----
    resid_train = train_payoffs .* (normalize_target ? y_std : 1) .+ (normalize_target ? y_mean : 0)
    resid_train .-= yhat_train
    resid_test  = test_payoffs .- yhat_test

    r2_train = 1 - sum(resid_train.^2) / sum(((train_payoffs .* (normalize_target ? y_std : 1) .+ (normalize_target ? y_mean : 0)) .- mean(train_payoffs .* (normalize_target ? y_std : 1) .+ (normalize_target ? y_mean : 0))).^2)
    r2_test  = 1 - sum(resid_test.^2)  / sum((test_payoffs .- mean(test_payoffs)).^2)
    rmse_train = sqrt(mean(resid_train.^2))
    rmse_test  = sqrt(mean(resid_test.^2))
    mae_train  = mean(abs.(resid_train))
    mae_test   = mean(abs.(resid_test))

    # ---- Coefficients (for reference) ----
    coeffs = if regression_method == :ols
        GLM.coef(model)[2:end]               # drop intercept
    else
        if lambda === nothing
            model.path.betas[:, argmin(model.meanloss)]
        else
            idx = argmin(abs.(model.lambda .- lambda))
            model.betas[:, idx]
        end
    end

    return (
        model = model,
        train_payoffs = normalize_target ? (train_payoffs .* y_std .+ y_mean) : train_payoffs,  # return in original units
        test_payoffs = test_payoffs,
        train_predictions = yhat_train,
        test_predictions = yhat_test,
        test_underlying_values = [underlying_func(p) for p in test_paths_2d],
        train_features = train_X,
        test_features = test_X,
        feature_names = kept_names,
        all_feature_names = all_feature_names,
        n_features_removed = length(all_feature_names) - length(kept_names),
        coefficients = coeffs,
        train_r2 = r2_train,
        test_r2 = r2_test,
        train_rmse = rmse_train,
        test_rmse = rmse_test,
        train_mae = mae_train,
        test_mae = mae_test,
    )
end


# ------------------------------------------------------------
# Plot multiple signature levels
# ------------------------------------------------------------

"""
    plot_signature_regressions(sig_levels; kwargs...) -> plt

Run `call_option_regression_experiment` for each `signature_level` in `sig_levels`
and plot predicted vs. true payoffs for each level on the same graph.

Example:
    plt = plot_signature_regressions([5, 6]; n_train=10000, n_test=1000)

Keyword args (forwarded to the experiment):
    n_train=10_000, n_test=1_000, S0=1.0, K=1.0, horizon=1.0, n_steps=100,
    μ=0.0, σ=0.7, verbose=false, underlying_func, payoff_func
"""
function plot_signature_regressions(sig_levels::AbstractVector{<:Integer};
    n_train::Int = 10_000,
    n_test::Int = 1_000,
    S0::Float64 = 1.0,
    K::Float64 = 1.0,
    horizon::Float64 = 1.0,
    n_steps::Int = 100,
    μ::Float64 = 0.0,
    σ::Float64 = 0.7,
    verbose::Bool = false,
    underlying_func = path -> path[end][2],                    # S_T (Euro)
    payoff_func     = (path, K) -> max(underlying_func(path) - K, 0.0),
    path_transform::Function = identity,
)
    # Helper to set reasonable axis limits
    autoscale_range(v; pad=0.05) = begin
        q1, q2 = quantile(v, (0.01, 0.99))
        span   = max(eps(), q2 - q1)
        (q1 - pad*span, q2 + pad*span)
    end

    # Predefine markers/colors for distinction
    shapes = [:circle, :utriangle, :rect, :diamond, :star5, :cross, :pentagon]
    cols   = [:royalblue, :purple, :seagreen, :darkorange, :crimson, :goldenrod, :teal]

    # Run and collect results
    results = Dict{Int, NamedTuple}()
    for (i, L) in enumerate(sig_levels)
        params = (
            n_train=n_train, n_test=n_test, S0=S0, K=K, horizon=horizon,
            n_steps=n_steps, μ=μ, σ=σ, verbose=verbose,
            underlying_func=underlying_func, payoff_func=payoff_func,
            signature_level=L,
            path_transform=path_transform,
        )
        res = call_option_regression_experiment(; params..., regression_method=:ols)
        results[L] = (
            x = res.test_underlying_values,
            y_true = res.test_payoffs,
            y_pred = res.test_predictions,
        )
    end

    # Combine all data for scaling
    all_x = reduce(vcat, [r.x for r in values(results)])
    all_y = reduce(vcat, [vcat(r.y_true, r.y_pred) for r in values(results)])
    xlims = autoscale_range(all_x)
    ylims = autoscale_range(all_y)

    # Create figure (small, crisp markers; no jitter)
    plt = plot(
        size=(950, 650), dpi=300, legend=:topright, grid=:on,
        framestyle=:box, xlim=xlims, ylim=ylims,
        xlabel="Underlying value", ylabel="Payoff / Prediction",
        title="Signature Regression vs True Payoff"
    )

    for (i, L) in enumerate(sig_levels)
        color = cols[mod1(i, length(cols))]
        shape = shapes[mod1(i, length(shapes))]

        scatter!(
            plt,
            results[L].x, results[L].y_pred;
            label = "Prediction (Level $L)",
            markershape = shape,
            markersize = 2.0,
            markercolor = color,      # <- pass the Symbol directly
            seriesalpha = 0.40,       # <- handles transparency
            markerstrokecolor = :black,
            markerstrokewidth = 0.1,
        )
    end


    # Add true payoff (use the first level as reference)
    ref_level = first(sig_levels)
    scatter!(
        plt,
        results[ref_level].x, results[ref_level].y_true;
        label = "True payoff",
        markershape = :diamond,
        markersize = 1.8,
        markercolor = :black,
        seriesalpha = 0.85,
    )

    return plt
end

# ------------------------------------------------------------
# Example usage (uncomment to run)
# ------------------------------------------------------------

path_transform = identity
underlying_euro(log_path)  = exp(log_path[end][2])
payoff_call(log_path, K)   = max(underlying_euro(log_path) - K, 0.0)
plt = plot_signature_regressions([5,6];
    n_train=10_000, n_test=1_000, μ=0.0, σ=0.001, horizon=1.0, n_steps=100,
    underlying_func = underlying_euro,
    payoff_func = payoff_call,
    path_transform = path_transform
)
savefig(plt, "regression_plot_levels_5_6_old.png")
display(plt)
