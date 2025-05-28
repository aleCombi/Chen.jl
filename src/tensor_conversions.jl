"""
Convert between plain signature vectors and SymbolicTensor format
"""

function vector_to_tensor(sig_vector::Vector{Float64}, word_map::Vector{Word}, dim::Int=2)
    """Convert signature vector to SymbolicTensor"""
    coeffs = Dict{Word, Float64}()
    for (i, word) in enumerate(word_map)
        if i <= length(sig_vector)
            coeffs[word] = sig_vector[i]
        end
    end
    return SymbolicTensor{Float64}(coeffs, dim)
end

function tensor_to_vector(σ::SymbolicTensor, word_map::Vector{Word})
    """Convert SymbolicTensor to vector using word ordering"""
    sig_vector = zeros(Float64, length(word_map))
    for (i, word) in enumerate(word_map)
        sig_vector[i] = σ[word]  # Uses getindex, returns 0 if not found
    end
    return sig_vector
end

function lexicographic_word_map(max_order::Int, dim::Int=2)
    """Generate lexicographic word ordering using existing signature_words functions"""
    words = Word[]
    
    # Order 0: ∅
    push!(words, Word())
    
    # Orders 1 to max_order
    for order in 1:max_order
        for indices in signature_words(order, dim)
            push!(words, Word(collect(indices)))
        end
    end
    
    return words
end

# Alias for default
standard_word_map(max_order::Int, dim::Int=2) = lexicographic_word_map(max_order, dim)

export vector_to_tensor, tensor_to_vector, standard_word_map