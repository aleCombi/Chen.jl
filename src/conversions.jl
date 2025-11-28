function SparseTensor(t::Tensor{T,D,M}) where {T,D,M}
    s = t.offsets
    coeffs = Dict{Word, T}()
    
    c0 = t.coeffs[s[1]+1]
    !iszero(c0) && (coeffs[Word()] = c0)
    
    len = 1
    @inbounds for k in 1:M
        start = s[k+1] + 1
        len *= D
        for p in 1:len
            c = t.coeffs[start + p - 1]
            if !iszero(c)
                rem = p - 1
                idxs = Vector{Int}(undef, k)
                base = (D == 1 ? 1 : div(len, D))
                for j in 1:k
                    q, rem = divrem(rem, base)
                    idxs[j] = q + 1
                    base = (D == 1 ? 1 : div(base, D))
                end
                coeffs[Word(idxs)] = c
            end
        end
    end
    return SparseTensor{T}(coeffs, D, M)
end

function Tensor(t::SparseTensor{T}) where {T}
    return _sparse_to_dense(t, Val(t.dim), Val(t.level))
end

function _sparse_to_dense(t::SparseTensor{T}, ::Val{D}, ::Val{M}) where {T,D,M}
    out = Tensor{T,D,M}()
    s = out.offsets
    
    for (w, c) in t.coeffs
        k = length(w)
        if k == 0
            out.coeffs[s[1] + 1] = c
        elseif k <= M
            pos = 0
            for j in 1:k
                pos = pos * D + (w.indices[j] - 1)
            end
            out.coeffs[s[k+1] + pos + 1] = c
        end
    end
    return out
end