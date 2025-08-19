using LinearAlgebra

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

Base.iterate(w::Word) = length(w) == 0 ? nothing : (w[1], 2)
Base.iterate(w::Word, i::Int) = i > length(w) ? nothing : (w[i], i+1)

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

Represents an element of the (extended) tensor algebra TN((ℝᵈ)) as a sparse 
collection of coefficients indexed by words.

Fields:
- `coeffs`: Dictionary mapping words to their coefficients
- `dim`: Dimension of the base space ℝᵈ
- `level`: Maximum tensor order (nothing for infinite)
"""
struct SparseTensor{T} <: AbstractTensor{T}
    coeffs::Dict{Word, T}
    dim::Int
    level::Int
    
    function SparseTensor{T}(coeffs::Dict{Word, T}, dim::Int, level::Int) where T
        # Filter out zero coefficients
        filtered_coeffs = filter(p -> !iszero(p.second), coeffs)
        new{T}(filtered_coeffs, dim, level)
    end
end

unit!(t::SparseTensor{T}) where {T} = (empty!(t.coeffs); t.coeffs[Word()] = one(T); t)

dim(ts::SparseTensor)   = ts.dim
level(ts::SparseTensor) = ts.level

function lift1!(st::SparseTensor{T}, x::AbstractVector{T}) where {T}
    @assert length(x) == st.dim
    empty!(st.coeffs)
    @inbounds for i in eachindex(x)
        v = x[i]
        if !iszero(v)
            st.coeffs[Word(i)] = v
        end
    end
    return st
end

# Convenient constructors
SparseTensor(coeffs::Dict{Word, T}, dim::Int, level::Int) where T = 
    SparseTensor{T}(coeffs, dim, level)

SparseTensor{T}(dim::Int, level::Int) where T = 
    SparseTensor{T}(Dict{Word, T}(), dim, level)

    @inline function Base.copy!(dest::SparseTensor{T}, src::SparseTensor{T}) where {T}
    # clear destination coefficients but keep capacity
    empty!(dest.coeffs)

    # copy all entries
    for (w, c) in src.coeffs
        dest.coeffs[w] = c
    end

    dest
end

Base.similar(ts::SparseTensor{T}) where {T} = SparseTensor{T}(ts.dim, ts.level)

# Zero tensor
Base.zero(::Type{SparseTensor{T}}, dim::Int, level::Int) where T = 
    SparseTensor{T}(dim, level)

Base.zero(t::SparseTensor{T}) where T = zero(SparseTensor{T}, t.dim, t.level)

@inline _write_unit!(t::PathSignatures.SparseTensor{T}) where {T} =
    (t.coeffs[Word()] = one(T); t)

# Coefficient access
function Base.getindex(t::SparseTensor{T}, w::Word) where T
    get(t.coeffs, w, zero(T))
end

function Base.setindex!(t::SparseTensor{T}, val::T, w::Word) where T
    if iszero(val)
        delete!(t.coeffs, w)
    else
        # Check level constraint
        if t.level !== nothing && length(w) > t.level
            throw(ArgumentError("Word length $(length(w)) exceeds level $(t.level)"))
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
Base.iterate(t::SparseTensor) = iterate(t.coeffs)
Base.iterate(t::SparseTensor, state) = iterate(t.coeffs, state)
Base.length(t::SparseTensor) = length(t.coeffs)

# Display
function Base.show(io::IO, t::SparseTensor{T}) where T
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
function Base.:+(t1::SparseTensor{T}, t2::SparseTensor{T}) where T
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
    
    level = something(t1.level, t2.level)
    if t1.level !== nothing && t2.level !== nothing
        level = max(t1.level, t2.level)
    end
    
    return SparseTensor{T}(result_coeffs, t1.dim, level)
end

function Base.:-(t1::SparseTensor{T}, t2::SparseTensor{T}) where T
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
    
    level = something(t1.level, t2.level)
    if t1.level !== nothing && t2.level !== nothing
        level = max(t1.level, t2.level)
    end
    
    return SparseTensor{T}(result_coeffs, t1.dim, level)
end

function Base.:*(scalar, t::SparseTensor{T}) where T
    if iszero(scalar)
        return zero(t)
    end
    
    result_coeffs = Dict{Word, T}()
    for (word, coeff) in t.coeffs
        result_coeffs[word] = scalar * coeff
    end
    
    return SparseTensor{T}(result_coeffs, t.dim, t.level)
end

Base.:*(t::SparseTensor, scalar) = scalar * t

# Tensor product operation
function mul!(dest::SparseTensor{T},
                   a::SparseTensor{T},
                   b::SparseTensor{T}) where {T}

    empty!(dest.coeffs)  # clear previous contents

    for (wa, ca) in a.coeffs
        for (wb, cb) in b.coeffs
            len = length(wa) + length(wb)
            if len <= dest.level
                key = wa * wb    
                dest.coeffs[key] = get(dest.coeffs, key, zero(T)) + ca * cb
            end
        end
    end
    return dest
end


"""
    shuffle_product!(out, t1, t2)

In-place shuffle product. Writes the result of `t1 ⨂ t2` into `out`.
`out` must be a `SparseTensor` with compatible `dim` and will have its coeffs cleared.
"""
function shuffle_product!(out::SparseTensor{T}, t1::SparseTensor{T}, t2::SparseTensor{T}) where {T}
    @assert t1.dim == t2.dim "Tensor dimensions must match"
    @assert out.dim == t1.dim "Output tensor dimension mismatch"

    empty!(out.coeffs)

    for (w1, c1) in t1.coeffs
        for (w2, c2) in t2.coeffs
            shuffles = compute_shuffles(w1, w2)
            coeff = c1 * c2

            for shuffle_word in shuffles
                # level check
                level = something(t1.level, t2.level)
                if t1.level !== nothing && t2.level !== nothing
                    level = max(t1.level, t2.level)
                end

                if length(shuffle_word) <= level
                    out.coeffs[shuffle_word] = get(out.coeffs, shuffle_word, zero(T)) + coeff
                end
            end
        end
    end

    out.level = if t1.level !== nothing && t2.level !== nothing
        max(t1.level, t2.level)
    else
        something(t1.level, t2.level)
    end

    return out
end

"""
    shuffle_product(t1, t2) -> SparseTensor

Allocating shuffle product. Creates a fresh `SparseTensor`.
"""
function shuffle_product(t1::SparseTensor{T}, t2::SparseTensor{T}) where {T}
    return shuffle_product!(similar(t1), t1, t2)
end

# --- operator alias (⊔ = shuffle product) ---
const ⊔ = shuffle_product

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
function projection(t::SparseTensor{T}, u::Word) where T
    result_coeffs = Dict{Word, T}()
    
    for (word, coeff) in t.coeffs
        if length(word) >= length(u) && word.indices[end-length(u)+1:end] == u.indices
            new_word = Word(word.indices[1:end-length(u)])
            result_coeffs[new_word] = coeff
        end
    end
    
    return SparseTensor{T}(result_coeffs, t.dim, t.level)
end

# Bracket operation ⟨ℓ, p⟩ from the paper
function bracket(ℓ::SparseTensor{T}, p::SparseTensor{T}) where T
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
function basis_element(::Type{T}, dim::Int, word::Word, level::Union{Int, Nothing}=nothing) where T
    coeffs = Dict{Word, T}(word => one(T))
    return SparseTensor{T}(coeffs, dim, level)
end

function empty_word_element(::Type{T}, dim::Int, level::Union{Int, Nothing}=nothing) where T
    return basis_element(T, dim, Word(), level)
end

function single_letter_element(::Type{T}, dim::Int, i::Int, level::Union{Int, Nothing}=nothing) where T
    return basis_element(T, dim, Word(i), level)
end

# Truncation operation
function truncate(t::SparseTensor{T}, level::Int) where T
    result_coeffs = Dict{Word, T}()
    
    for (word, coeff) in t.coeffs
        if length(word) <= level
            result_coeffs[word] = coeff
        end
    end
    
    return SparseTensor{T}(result_coeffs, t.dim, level)
end

"""
    shuffle_exponential(ℓ::SymbolicTensor{T}) -> SymbolicTensor{T}

Compute the shuffle exponential e^⊔⊔ℓ = ∑_{n=0}^∞ ℓ^⊔⊔n/n! (Equation 2.7)
"""
function shuffle_exponential(ℓ::SparseTensor{T}; max_terms::Int=10) where T
    result = empty_word_element(T, ℓ.dim, ℓ.level)
    ℓ_power = empty_word_element(T, ℓ.dim, ℓ.level)  # ℓ^⊔⊔0 = ∅
    
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
function resolvent(ℓ::SparseTensor{T}; max_terms::Int=10) where T
    # Check that ℓ_∅ = 0
    empty_word = Word()
    if !iszero(ℓ[empty_word])
        throw(ArgumentError("For resolvent to be well-defined, coefficient of empty word must be zero"))
    end
    
    result = empty_word_element(T, ℓ.dim, ℓ.level)
    ℓ_power = empty_word_element(T, ℓ.dim, ℓ.level)  # ℓ^⊗0 = ∅
    
    for n in 0:max_terms
        result = result + ℓ_power
        
        if n < max_terms
            ℓ_power = tensor_product(ℓ_power, ℓ)
        end
    end
    
    return result
end

@inline function _zero!(ts::SparseTensor{T}) where {T}
    empty!(ts.coeffs)    # mutate in place
    return ts
end

@inline function add_scaled!(dest::SparseTensor{T}, src::SparseTensor{T}, α::T) where {T}
    for (w, c) in src.coeffs
        newv = get(dest.coeffs, w, zero(T)) + α * c
        if iszero(newv)
            delete!(dest.coeffs, w)
        else
            dest.coeffs[w] = newv
        end
    end
    return dest
end

# Export main types and functions
export SparseTensor, Word, AbstractTensorAlgebra
export tensor_product, shuffle_product, projection, bracket
export basis_element, empty_word_element, single_letter_element, truncate
export resolvent, shuffle_exponential