using LinearAlgebra

"""
    Word
Represents a word (multi-index) in the tensor algebra basis.
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
Base.:*(w1::Word, w2::Word) = Word(vcat(w1.indices, w2.indices))

"""
    SparseTensor{T}
Represents an element of the (extended) tensor algebra.
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
SparseTensor(coeffs::Dict{Word, T}, dim::Int, level::Int) where T = SparseTensor{T}(coeffs, dim, level)
SparseTensor{T}(dim::Int, level::Int) where T = SparseTensor{T}(Dict{Word, T}(), dim, level)

@inline function Base.copy!(dest::SparseTensor{T}, src::SparseTensor{T}) where {T}
    empty!(dest.coeffs)
    for (w, c) in src.coeffs
        dest.coeffs[w] = c
    end
    dest
end

# -------- Sparse ↔ Sparse --------
# FIXED: Removed `Chen.` prefix
function Base.isapprox(A::SparseTensor{Ta}, B::SparseTensor{Tb};
                       atol::Real=1e-8, rtol::Real=1e-8) where {Ta,Tb}
    A.dim == B.dim && A.level == B.level || return false
    RA = promote_type(Ta, Tb)
    aC, bC = A.coeffs, B.coeffs
    @inbounds for (w, va) in aC
        vb = haskey(bC, w) ? bC[w] : zero(RA)
        vaR = RA(va); vbR = RA(vb)
        if !(abs(vaR - vbR) <= atol + rtol*max(abs(vaR),abs(vbR))); return false; end
    end
    @inbounds for (w, vb) in bC
        if !haskey(aC, w)
            vbR = RA(vb); vaR = zero(RA)
            if !(abs(vaR - vbR) <= atol + rtol*max(abs(vaR),abs(vbR))); return false; end
        end
    end
    return true
end

Base.similar(ts::SparseTensor{T}) where {T} = SparseTensor{T}(ts.dim, ts.level)
Base.zero(::Type{SparseTensor{T}}, dim::Int, level::Int) where T = SparseTensor{T}(dim, level)
Base.zero(t::SparseTensor{T}) where T = zero(SparseTensor{T}, t.dim, t.level)

# FIXED: Removed `Chen.` prefix
@inline _write_unit!(t::SparseTensor{T}) where {T} = (t.coeffs[Word()] = one(T); t)

# Coefficient access
Base.getindex(t::SparseTensor{T}, w::Word) where T = get(t.coeffs, w, zero(T))

function Base.setindex!(t::SparseTensor{T}, val::T, w::Word) where T
    if iszero(val)
        delete!(t.coeffs, w)
    else
        if t.level !== nothing && length(w) > t.level
            throw(ArgumentError("Word length $(length(w)) exceeds level $(t.level)"))
        end
        if any(i -> i < 1 || i > t.dim, w.indices)
            throw(ArgumentError("Word indices must be in range 1:$(t.dim)"))
        end
        t.coeffs[w] = val
    end
    return t
end

Base.iterate(t::SparseTensor) = iterate(t.coeffs)
Base.iterate(t::SparseTensor, state) = iterate(t.coeffs, state)
Base.length(t::SparseTensor) = length(t.coeffs)

function Base.show(io::IO, t::SparseTensor{T}) where T
    if isempty(t.coeffs)
        print(io, "0")
        return
    end
    first_term = true
    for (word, coeff) in sort(collect(t.coeffs), by=p->length(p.first.indices))
        if !first_term; print(io, " + "); end
        first_term = false
        if coeff == 1 && !isempty(word.indices)
            print(io, word)
        elseif coeff == -1 && !isempty(word.indices)
            print(io, "-", word)
        else
            print(io, coeff)
            if !isempty(word.indices); print(io, "⋅", word); end
        end
    end
end

# Arithmetic
function Base.:+(t1::SparseTensor{T}, t2::SparseTensor{T}) where T
    @assert t1.dim == t2.dim
    result_coeffs = copy(t1.coeffs)
    for (word, coeff) in t2.coeffs
        new_coeff = get(result_coeffs, word, zero(T)) + coeff
        iszero(new_coeff) ? delete!(result_coeffs, word) : (result_coeffs[word] = new_coeff)
    end
    level = max(t1.level, t2.level)
    return SparseTensor{T}(result_coeffs, t1.dim, level)
end

function Base.:-(t1::SparseTensor{T}, t2::SparseTensor{T}) where T
    @assert t1.dim == t2.dim
    result_coeffs = copy(t1.coeffs)
    for (word, coeff) in t2.coeffs
        new_coeff = get(result_coeffs, word, zero(T)) - coeff
        iszero(new_coeff) ? delete!(result_coeffs, word) : (result_coeffs[word] = new_coeff)
    end
    level = max(t1.level, t2.level)
    return SparseTensor{T}(result_coeffs, t1.dim, level)
end

function Base.:*(scalar, t::SparseTensor{T}) where T
    iszero(scalar) && return zero(t)
    result_coeffs = Dict{Word, T}(w => scalar * c for (w, c) in t.coeffs)
    return SparseTensor{T}(result_coeffs, t.dim, t.level)
end
Base.:*(t::SparseTensor, scalar) = scalar * t

function mul!(dest::SparseTensor{T}, a::SparseTensor{T}, b::SparseTensor{T}) where {T}
    empty!(dest.coeffs)
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

function shuffle_product!(out::SparseTensor{T}, t1::SparseTensor{T}, t2::SparseTensor{T}) where {T}
    @assert t1.dim == t2.dim
    @assert out.dim == t1.dim
    empty!(out.coeffs)
    for (w1, c1) in t1.coeffs
        for (w2, c2) in t2.coeffs
            coeff = c1 * c2
            for shuffle_word in compute_shuffles(w1, w2)
                if length(shuffle_word) <= out.level
                    out.coeffs[shuffle_word] = get(out.coeffs, shuffle_word, zero(T)) + coeff
                end
            end
        end
    end
    return out
end

function shuffle_product(t1::SparseTensor{T}, t2::SparseTensor{T}) where {T}
    return shuffle_product!(similar(t1), t1, t2)
end

const ⊔ = shuffle_product

function compute_shuffles(w1::Word, w2::Word)
    if isempty(w1.indices); return [w2]; end
    if isempty(w2.indices); return [w1]; end
    result = Word[]
    # 1. Take from w1
    for s in compute_shuffles(Word(w1.indices[2:end]), w2)
        push!(result, Word([w1.indices[1]; s.indices]))
    end
    # 2. Take from w2
    for s in compute_shuffles(w1, Word(w2.indices[2:end]))
        push!(result, Word([w2.indices[1]; s.indices]))
    end
    return result
end

function bracket(ℓ::SparseTensor{T}, p::SparseTensor{T}) where T
    @assert ℓ.dim == p.dim
    result = zero(T)
    for (word, coeff_ℓ) in ℓ.coeffs
        if haskey(p.coeffs, word)
            result += coeff_ℓ * p.coeffs[word]
        end
    end
    return result
end

function basis_element(::Type{T}, dim::Int, word::Word, level::Int) where T
    return SparseTensor{T}(Dict(word => one(T)), dim, level)
end

function single_letter_element(::Type{T}, dim::Int, i::Int, level::Int) where T
    return basis_element(T, dim, Word(i), level)
end

@inline function _zero!(ts::SparseTensor{T}) where {T}
    empty!(ts.coeffs)
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

export SparseTensor, Word, shuffle_product, bracket