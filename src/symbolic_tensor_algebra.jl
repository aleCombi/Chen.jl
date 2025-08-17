using LinearAlgebra

"""
    AbstractTensorAlgebra

Abstract base type for tensor algebra elements.
"""
abstract type AbstractTensorAlgebra end

"""
    Word

Represents a word (multi-index) in the tensor algebra basis.
A word is a sequence of indices i₁i₂...iₙ corresponding to eᵢ₁ ⊗ eᵢ₂ ⊗ ... ⊗ eᵢₙ
"""
struct Word
    indices::Vector{Int}
    
    Word(indices::Vector{Int}) = new(indices)
    Word(indices::Int...) = new(collect(indices))
    Word() = new(Int[])  # Empty word ∅
end

# Word operations
Base.length(w::Word) = length(w.indices)
Base.:(==)(w1::Word, w2::Word) = w1.indices == w2.indices
Base.hash(w::Word, h::UInt) = hash(w.indices, h)
Base.show(io::IO, w::Word) = isempty(w.indices) ? print(io, "∅") : print(io, join(w.indices))

# Concatenation of words
function Base.:*(w1::Word, w2::Word)
    Word(vcat(w1.indices, w2.indices))
end

"""
    SymbolicTensor{T}

Represents an element of the (extended) tensor algebra T((ℝᵈ)) as a sparse 
collection of coefficients indexed by words.

Fields:
- `coeffs`: Dictionary mapping words to their coefficients
- `dim`: Dimension of the base space ℝᵈ
- `max_order`: Maximum tensor order (nothing for infinite)
"""
struct SymbolicTensor{T} <: AbstractTensorAlgebra
    coeffs::Dict{Word, T}
    dim::Int
    max_order::Union{Int, Nothing}
    
    function SymbolicTensor{T}(coeffs::Dict{Word, T}, dim::Int, max_order::Union{Int, Nothing}=nothing) where T
        # Filter out zero coefficients
        filtered_coeffs = filter(p -> !iszero(p.second), coeffs)
        new{T}(filtered_coeffs, dim, max_order)
    end
end

# Convenient constructors
SymbolicTensor(coeffs::Dict{Word, T}, dim::Int, max_order::Union{Int, Nothing}=nothing) where T = 
    SymbolicTensor{T}(coeffs, dim, max_order)

SymbolicTensor{T}(dim::Int, max_order::Union{Int, Nothing}=nothing) where T = 
    SymbolicTensor{T}(Dict{Word, T}(), dim, max_order)

# Zero tensor
Base.zero(::Type{SymbolicTensor{T}}, dim::Int, max_order::Union{Int, Nothing}=nothing) where T = 
    SymbolicTensor{T}(dim, max_order)

Base.zero(t::SymbolicTensor{T}) where T = zero(SymbolicTensor{T}, t.dim, t.max_order)

# Coefficient access
function Base.getindex(t::SymbolicTensor{T}, w::Word) where T
    get(t.coeffs, w, zero(T))
end

function Base.setindex!(t::SymbolicTensor{T}, val::T, w::Word) where T
    if iszero(val)
        delete!(t.coeffs, w)
    else
        # Check max_order constraint
        if t.max_order !== nothing && length(w) > t.max_order
            throw(ArgumentError("Word length $(length(w)) exceeds max_order $(t.max_order)"))
        end
        # Check dimension constraint
        if any(i -> i < 1 || i > t.dim, w.indices)
            throw(ArgumentError("Word indices must be in range 1:$(t.dim)"))
        end
        t.coeffs[w] = val
    end
    return t
end

# Iteration interface
Base.iterate(t::SymbolicTensor) = iterate(t.coeffs)
Base.iterate(t::SymbolicTensor, state) = iterate(t.coeffs, state)
Base.length(t::SymbolicTensor) = length(t.coeffs)

# Display
function Base.show(io::IO, t::SymbolicTensor{T}) where T
    if isempty(t.coeffs)
        print(io, "0")
        return
    end
    
    first_term = true
    for (word, coeff) in sort(collect(t.coeffs), by=p->length(p.first.indices))
        if !first_term
            print(io, " + ")
        end
        first_term = false
        
        if coeff == 1 && !isempty(word.indices)
            print(io, word)
        elseif coeff == -1 && !isempty(word.indices)
            print(io, "-", word)
        else
            print(io, coeff)
            if !isempty(word.indices)
                print(io, "⋅", word)
            end
        end
    end
end

# Basic arithmetic operations
function Base.:+(t1::SymbolicTensor{T}, t2::SymbolicTensor{T}) where T
    @assert t1.dim == t2.dim "Tensor dimensions must match"
    
    result_coeffs = copy(t1.coeffs)
    for (word, coeff) in t2.coeffs
        if haskey(result_coeffs, word)
            new_coeff = result_coeffs[word] + coeff
            if iszero(new_coeff)
                delete!(result_coeffs, word)
            else
                result_coeffs[word] = new_coeff
            end
        else
            result_coeffs[word] = coeff
        end
    end
    
    max_order = something(t1.max_order, t2.max_order)
    if t1.max_order !== nothing && t2.max_order !== nothing
        max_order = max(t1.max_order, t2.max_order)
    end
    
    return SymbolicTensor{T}(result_coeffs, t1.dim, max_order)
end

function Base.:-(t1::SymbolicTensor{T}, t2::SymbolicTensor{T}) where T
    @assert t1.dim == t2.dim "Tensor dimensions must match"
    
    result_coeffs = copy(t1.coeffs)
    for (word, coeff) in t2.coeffs
        if haskey(result_coeffs, word)
            new_coeff = result_coeffs[word] - coeff
            if iszero(new_coeff)
                delete!(result_coeffs, word)
            else
                result_coeffs[word] = new_coeff
            end
        else
            result_coeffs[word] = -coeff
        end
    end
    
    max_order = something(t1.max_order, t2.max_order)
    if t1.max_order !== nothing && t2.max_order !== nothing
        max_order = max(t1.max_order, t2.max_order)
    end
    
    return SymbolicTensor{T}(result_coeffs, t1.dim, max_order)
end

function Base.:*(scalar, t::SymbolicTensor{T}) where T
    if iszero(scalar)
        return zero(t)
    end
    
    result_coeffs = Dict{Word, T}()
    for (word, coeff) in t.coeffs
        result_coeffs[word] = scalar * coeff
    end
    
    return SymbolicTensor{T}(result_coeffs, t.dim, t.max_order)
end

Base.:*(t::SymbolicTensor, scalar) = scalar * t

# Tensor product operation
function tensor_product(t1::SymbolicTensor{T}, t2::SymbolicTensor{T}) where T
    @assert t1.dim == t2.dim "Tensor dimensions must match"
    
    result_coeffs = Dict{Word, T}()
    
    for (w1, c1) in t1.coeffs
        for (w2, c2) in t2.coeffs
            new_word = w1 * w2  # Concatenation
            coeff = c1 * c2
            
            # Check max_order constraint
            max_order = something(t1.max_order, t2.max_order)
            if t1.max_order !== nothing && t2.max_order !== nothing
                max_order = t1.max_order + t2.max_order
            end
            
            if max_order === nothing || length(new_word) <= max_order
                if haskey(result_coeffs, new_word)
                    result_coeffs[new_word] += coeff
                else
                    result_coeffs[new_word] = coeff
                end
            end
        end
    end
    
    max_order = something(t1.max_order, t2.max_order)
    if t1.max_order !== nothing && t2.max_order !== nothing
        max_order = t1.max_order + t2.max_order
    end
    
    return SymbolicTensor{T}(result_coeffs, t1.dim, max_order)
end

# Shuffle product (key operation from the paper)
function shuffle_product(t1::SymbolicTensor{T}, t2::SymbolicTensor{T}) where T
    @assert t1.dim == t2.dim "Tensor dimensions must match"
    
    result_coeffs = Dict{Word, T}()
    
    for (w1, c1) in t1.coeffs
        for (w2, c2) in t2.coeffs
            # Compute all shuffles of w1 and w2
            shuffles = compute_shuffles(w1, w2)
            coeff = c1 * c2
            
            for shuffle_word in shuffles
                # Check max_order constraint
                max_order = something(t1.max_order, t2.max_order)
                if t1.max_order !== nothing && t2.max_order !== nothing
                    max_order = max(t1.max_order, t2.max_order)
                end
                
                if max_order === nothing || length(shuffle_word) <= max_order
                    if haskey(result_coeffs, shuffle_word)
                        result_coeffs[shuffle_word] += coeff
                    else
                        result_coeffs[shuffle_word] = coeff
                    end
                end
            end
        end
    end
    
    max_order = something(t1.max_order, t2.max_order)
    if t1.max_order !== nothing && t2.max_order !== nothing
        max_order = max(t1.max_order, t2.max_order)
    end
    
    return SymbolicTensor{T}(result_coeffs, t1.dim, max_order)
end

# Helper function to compute all shuffles of two words
function compute_shuffles(w1::Word, w2::Word)
    if isempty(w1.indices)
        return [w2]
    elseif isempty(w2.indices)
        return [w1]
    else
        result = Word[]
        
        # Take first element from w1
        rest_w1 = Word(w1.indices[2:end])
        for shuffle in compute_shuffles(rest_w1, w2)
            push!(result, Word([w1.indices[1]; shuffle.indices]))
        end
        
        # Take first element from w2
        rest_w2 = Word(w2.indices[2:end])
        for shuffle in compute_shuffles(w1, rest_w2)
            push!(result, Word([w2.indices[1]; shuffle.indices]))
        end
        
        return result
    end
end

# Projection operation |_u from the paper
function projection(t::SymbolicTensor{T}, u::Word) where T
    result_coeffs = Dict{Word, T}()
    
    for (word, coeff) in t.coeffs
        if length(word) >= length(u) && word.indices[end-length(u)+1:end] == u.indices
            new_word = Word(word.indices[1:end-length(u)])
            result_coeffs[new_word] = coeff
        end
    end
    
    return SymbolicTensor{T}(result_coeffs, t.dim, t.max_order)
end

# Bracket operation ⟨ℓ, p⟩ from the paper
function bracket(ℓ::SymbolicTensor{T}, p::SymbolicTensor{T}) where T
    @assert ℓ.dim == p.dim "Tensor dimensions must match"
    
    result = zero(T)
    for (word, coeff_ℓ) in ℓ.coeffs
        if haskey(p.coeffs, word)
            result += coeff_ℓ * p.coeffs[word]
        end
    end
    
    return result
end

# Utility functions for creating basis elements
function basis_element(::Type{T}, dim::Int, word::Word, max_order::Union{Int, Nothing}=nothing) where T
    coeffs = Dict{Word, T}(word => one(T))
    return SymbolicTensor{T}(coeffs, dim, max_order)
end

function empty_word_element(::Type{T}, dim::Int, max_order::Union{Int, Nothing}=nothing) where T
    return basis_element(T, dim, Word(), max_order)
end

function single_letter_element(::Type{T}, dim::Int, i::Int, max_order::Union{Int, Nothing}=nothing) where T
    return basis_element(T, dim, Word(i), max_order)
end

# Truncation operation
function truncate(t::SymbolicTensor{T}, max_order::Int) where T
    result_coeffs = Dict{Word, T}()
    
    for (word, coeff) in t.coeffs
        if length(word) <= max_order
            result_coeffs[word] = coeff
        end
    end
    
    return SymbolicTensor{T}(result_coeffs, t.dim, max_order)
end

"""
    shuffle_exponential(ℓ::SymbolicTensor{T}) -> SymbolicTensor{T}

Compute the shuffle exponential e^⊔⊔ℓ = ∑_{n=0}^∞ ℓ^⊔⊔n/n! (Equation 2.7)
"""
function shuffle_exponential(ℓ::SymbolicTensor{T}; max_terms::Int=10) where T
    result = empty_word_element(T, ℓ.dim, ℓ.max_order)
    ℓ_power = empty_word_element(T, ℓ.dim, ℓ.max_order)  # ℓ^⊔⊔0 = ∅
    
    for n in 0:max_terms
        factorial_n = factorial(n)
        result = result + (1/factorial_n) * ℓ_power
        
        if n < max_terms
            ℓ_power = shuffle_product(ℓ_power, ℓ)
        end
    end
    
    return result
end

"""
    resolvent(ℓ::SymbolicTensor{T}) -> SymbolicTensor{T}

Compute the resolvent (∅ - ℓ)^{-1} = ∑_{n=0}^∞ ℓ^⊗n (Equation 2.5)
Assumes ℓ_∅ = 0
"""
function resolvent(ℓ::SymbolicTensor{T}; max_terms::Int=10) where T
    # Check that ℓ_∅ = 0
    empty_word = Word()
    if !iszero(ℓ[empty_word])
        throw(ArgumentError("For resolvent to be well-defined, coefficient of empty word must be zero"))
    end
    
    result = empty_word_element(T, ℓ.dim, ℓ.max_order)
    ℓ_power = empty_word_element(T, ℓ.dim, ℓ.max_order)  # ℓ^⊗0 = ∅
    
    for n in 0:max_terms
        result = result + ℓ_power
        
        if n < max_terms
            ℓ_power = tensor_product(ℓ_power, ℓ)
        end
    end
    
    return result
end

# Export main types and functions
export SymbolicTensor, Word, AbstractTensorAlgebra
export tensor_product, shuffle_product, projection, bracket
export basis_element, empty_word_element, single_letter_element, truncate
export resolvent, shuffle_exponential