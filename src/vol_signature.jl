"""
Ornstein-Uhlenbeck Process Signature Representations from Lemma 3.1

The OU process: dXₜ = κ(θ - Xₜ)dt + ηdWₜ, X₀ = x
"""

"""
Time-independent representation (Equation 3.8):
Xₜ = ⟨ℓᴼᵁ, W̃ₜᶜ⟩ where ℓᴼᵁ = (x∅ + κθ1 + η2)e^⊔⊔(-κ1)
"""
function ornstein_uhlenbeck_time_independent(x, κ, θ, η, dim::Int=2; max_order::Int=5)
    T = typeof(x)
    
    # Basis elements: ∅, 1 (time), 2 (Brownian motion)
    empty_elem = empty_word_element(T, dim, max_order)
    elem_1 = single_letter_element(T, dim, 1, max_order)  
    elem_2 = single_letter_element(T, dim, 2, max_order)  
    
    # Base term: (x∅ + κθ1 + η2)
    base_term = x * empty_elem + κ * θ * elem_1 + η * elem_2
    
    # Exponential term: e^⊔⊔(-κ1)
    neg_kappa_1 = (-κ) * elem_1
    exp_term = shuffle_exponential(neg_kappa_1, max_terms=max_order)
    
    # Final representation: base ⊔⊔ exp
    return shuffle_product(base_term, exp_term)
end

"""
Time-dependent representation (Equation 3.9):
Xₜ = ⟨ℓ̃ₜᴼᵁ, W̃ₜᶜ⟩ where ℓ̃ₜᴼᵁ = θ∅ + e^(-κt)[(x-θ)∅ + ηe^⊔⊔(κt2)]
"""
function ornstein_uhlenbeck_time_dependent(x, κ, θ, η, t, dim::Int=2; max_order::Int=5)
    T = typeof(x)
    
    # Basis elements
    empty_elem = empty_word_element(T, dim, max_order)
    elem_1 = single_letter_element(T, dim, 1, max_order)
    elem_2 = single_letter_element(T, dim, 2, max_order)
    
    # First term: θ∅
    first_term = θ * empty_elem
    
    # Second term components
    exp_neg_kt = exp(-κ * t)  # e^(-κt)
    
    # Inner term: (x-θ)∅ + ηe^⊔⊔(κt2)
    inner_base = (x - θ) * empty_elem
    
    # e^⊔⊔(κt2)
    kappa_t_2 = (κ * t) * elem_2
    exp_shuffle = shuffle_exponential(kappa_t_2, max_terms=max_order)
    
    inner_exp = η * exp_shuffle
    inner_term = inner_base + inner_exp
    
    # Second term: e^(-κt) * inner_term
    second_term = exp_neg_kt * inner_term
    
    # Final result
    return first_term + second_term
end

# Example usage
function compare_ou_representations()
    # Parameters
    x, κ, θ, η = 1.0, 2.0, 0.5, 0.3
    t = 0.5
    
    println("=== Ornstein-Uhlenbeck Representations ===")
    
    # Time-independent
    σ_ti = ornstein_uhlenbeck_time_independent(x, κ, θ, η, 2, max_order=3)
    println("Time-independent:")
    println(σ_ti)
    
    # Time-dependent  
    σ_td = ornstein_uhlenbeck_time_dependent(x, κ, θ, η, t, 2, max_order=3)
    println("\nTime-dependent (t=$t):")
    println(σ_td)
    
    return σ_ti, σ_td
end

export ornstein_uhlenbeck_time_independent, ornstein_uhlenbeck_time_dependent, compare_ou_representations