# src/lyndon_basis.jl
using LinearAlgebra

function is_lyndon(w::Word)
    idxs = w.indices
    n = length(idxs)
    n == 0 && return false
    n == 1 && return true
    for i in 2:n
        if idxs[i:end] <= idxs
            return false
        end
    end
    return true
end

function lyndon_words(d::Int, N::Int)
    result = Word[]
    buffer = Vector{Int}()
    function build!(L)
        if length(buffer) == L
            w = Word(copy(buffer))
            is_lyndon(w) && push!(result, w)
            return
        end
        for a in 1:d
            push!(buffer, a); build!(L); pop!(buffer)
        end
    end
    for L in 1:N
        levelwords = Word[]
        build!(L)
        for w in result[end-length(levelwords)+1:end]; push!(levelwords, w); end
        sort!(levelwords; by = w -> w.indices)
        append!(result[1:end-length(levelwords)], levelwords)
    end
    return result
end

function _longest_lyndon_suffix(w::Word, lynds::Vector{Word})
    n = length(w)
    lyset = Set(lynds)
    for L in (n-1):-1:1
        sfx = Word(w.indices[end-L+1:end])
        if sfx in lyset; return sfx; end
    end
    error("No Lyndon suffix for $w")
end

# helper to read dense coeff
# FIXED: Chen.Tensor -> Tensor
@inline function _coeff_of_word(t::Tensor{T}, w::Word) where {T}
    d = t.dim
    k = length(w)
    start0 = t.offsets[k+1]
    pos1   = 1
    @inbounds for j in 1:k
        pos1 += (w.indices[j]-1) * d^(k-j)
    end
    return t.coeffs[start0 + pos1]
end

# This builds the L matrix using SPARSE logic (cleaner) but caches Dense results?
# Let's clean this up to use Sparse internally as discussed.
function build_L(d::Int, N::Int; T=Float64)
    lynds = lyndon_words(d, N)
    m = length(lynds)
    L = zeros(T, m, m)
    
    # Use SparseTensor for the symbolic calculation
    Φcache = Dict{Word, SparseTensor{T}}()

    for (j, w) in enumerate(lynds)
        if length(w) == 1
            L[j,j] = 1
            Φcache[w] = single_letter_element(T, d, w.indices[1], N)
        else
            v = _longest_lyndon_suffix(w, lynds)
            u = Word(w.indices[1:end-length(v)])

            Φu = Φcache[u]
            Φv = Φcache[v]
            
            # Compute Lie bracket: [u, v] = u⊗v - v⊗u
            # We use the generic bracket from sparse_tensors.jl
            tmp1 = similar(Φu); mul!(tmp1, Φu, Φv)
            tmp2 = similar(Φu); mul!(tmp2, Φv, Φu)
            Φw = tmp1 - tmp2
            
            # Read coeffs
            for (i, wi) in enumerate(lynds)
                L[i,j] = get(Φw.coeffs, wi, zero(T))
            end
            Φcache[w] = Φw
        end
    end
    return lynds, L, Φcache
end

# FIXED: Chen.Tensor -> Tensor
function project_to_lyndon(u_dense::Tensor{T}, lynds::Vector{Word}, L::Matrix{T}) where {T}
    m = length(lynds)
    u = zeros(T, m)
    @inbounds for i in 1:m
        u[i] = _coeff_of_word(u_dense, lynds[i])
    end
    ℓ = LowerTriangular(L) \ u
    return ℓ
end