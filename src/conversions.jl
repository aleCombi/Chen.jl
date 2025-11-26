# --- Dense → Sparse ---
# We extend the imported SparseTensor function
# FIXED: Qualify with Algebra. to silence the extension warning
function Algebra.SparseTensor(t::Tensor{T,D,M}) where {T,D,M}
    s = t.offsets
    coeffs = Dict{Word, T}()
    
    c0 = t.coeffs[s[1]+1]
    if !iszero(c0); coeffs[Word()] = c0; end
    
    len = 1
    @inbounds for k in 1:M
        start = s[k+1] + 1
        len *= D
        for p in 1:len
            c = t.coeffs[start + p - 1]
            if !iszero(c)
                rem  = p - 1
                idxs = Vector{Int}(undef, k)
                base = (D == 1 ? 1 : div(len, D))
                @inbounds for j in 1:k
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

# --- Sparse → Dense ---
# We extend the Chen.Tensor function
function Tensor(t::SparseTensor{T}) where {T}
    return _sparse_to_dense(t, Val(t.dim), Val(t.level))
end

@generated function _sparse_to_dense(t::SparseTensor{T}, ::Val{D}, ::Val{M}) where {T,D,M}
    quote
        out = Tensor{T,D,M}()
        fill!(out.coeffs, zero(T))
        s = out.offsets

        @inbounds for (w, c) in t.coeffs
            k = length(w)
            if k == 0
                out.coeffs[s[1] + 1] = c
            else
                posm1 = 0
                @inbounds for j in 1:k
                    posm1 = posm1 * $D + (w.indices[j] - 1)
                end
                idx = s[k + 1] + posm1 + 1
                out.coeffs[idx] = c
            end
        end
        return out
    end
end