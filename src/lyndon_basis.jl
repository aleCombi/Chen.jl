# check if a Word is Lyndon
function is_lyndon(w::Word)
    idxs = w.indices
    n = length(idxs)
    n == 0 && return false
    n == 1 && return true
    for i in 2:n
        if idxs[i:end] <= idxs   # vector lex compare
            return false
        end
    end
    return true
end

# enumerate all Lyndon words up to length N over alphabet 1:d
"""
    lyndon_words(d::Int, N::Int) -> Vector{Word}

Return all Lyndon words over alphabet 1:d of length ≤ N, in the
same order as iisignature: grouped by length (level), then
lexicographically within each length.
"""
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
            push!(buffer, a)
            build!(L)
            pop!(buffer)
        end
    end

    for L in 1:N
        levelwords = Word[]
        build!(L)                       # fills `result` with length-L Lyndon words
        # collect them
        for w in result[end-length(levelwords)+1:end]
            push!(levelwords, w)
        end
        # sort lex within the level
        sort!(levelwords; by = w -> w.indices)
        append!(result[1:end-length(levelwords)], levelwords)
    end

    return result
end

function bracket(u::Tensor{T}, v::Tensor{T}) where {T}
    tmp1 = similar(u)
    tmp2 = similar(u)
    mul!(tmp1, u, v)
    mul!(tmp2, v, u)
    tmp1.coeffs .-= tmp2.coeffs
    return tmp1
end

using LinearAlgebra

# one-hot tensor for a single letter
function _basis_tensor(::Type{T}, d::Int, N::Int, i::Int) where {T}
    t = Chen.Tensor{T}(d, N)
    fill!(t.coeffs, zero(T))
    start_lvl1 = t.offsets[1 + 1]      # level-1 block start (0-based)
    t.coeffs[start_lvl1 + i] = one(T)  # +i because indices are 1-based
    return t
end

# coefficient of tensor word w in dense tensor t
@inline function _coeff_of_word(t::Chen.Tensor{T}, w::Word) where {T}
    d = t.dim
    k = length(w)
    start0 = t.offsets[k+1]
    pos1   = 1
    @inbounds for j in 1:k
        pos1 += (w.indices[j]-1) * d^(k-j)
    end
    return t.coeffs[start0 + pos1]
end

# longest Lyndon suffix
function _longest_lyndon_suffix(w::Word, lynds::Vector{Word})
    n = length(w)
    lyset = Set(lynds)
    for L in (n-1):-1:1
        sfx = Word(w.indices[end-L+1:end])
        if sfx in lyset
            return sfx
        end
    end
    error("No Lyndon suffix for $w")
end

"""
    build_L(d, N; T=Float64) -> (lynds, L, Φcache)

Construct Lyndon words and the lower-triangular matrix L (tensor->Lyndon).
Also returns a cache of dense tensors Φ(w).
"""
function build_L(d::Int, N::Int; T=Float64)
    lynds = lyndon_words(d, N)         # you already have this
    m = length(lynds)
    L = zeros(T, m, m)
    Φcache = Dict{Word, Chen.Tensor{T}}()

    for (j, w) in enumerate(lynds)
        if length(w) == 1
            L[j,j] = 1
            Φcache[w] = _basis_tensor(T, d, N, w.indices[1])
        else
            v = _longest_lyndon_suffix(w, lynds)
            u = Word(w.indices[1:end-length(v)])

            Φu = Φcache[u]
            Φv = Φcache[v]

            tmp1 = Chen.Tensor{T}(d, N)
            tmp2 = Chen.Tensor{T}(d, N)
            Φw   = Chen.Tensor{T}(d, N)

            mul!(tmp1, Φu, Φv)
            mul!(tmp2, Φv, Φu)
            @. Φw.coeffs = tmp1.coeffs - tmp2.coeffs

            for (i, wi) in enumerate(lynds)
                L[i,j] = _coeff_of_word(Φw, wi)
            end

            Φcache[w] = Φw
        end
    end
    return lynds, L, Φcache
end

"""
    project_to_lyndon(u_dense, lynds, L)

Project a dense log-signature (tensor basis) onto Lyndon coordinates.
"""
function project_to_lyndon(u_dense::Chen.Tensor{T},
                           lynds::Vector{Word},
                           L::Matrix{T}) where {T}
    m = length(lynds)
    u = zeros(T, m)
    @inbounds for i in 1:m
        u[i] = _coeff_of_word(u_dense, lynds[i])
    end
    ℓ = LowerTriangular(L) \ u
    return ℓ
end
