# src/conversions.jl

# --- Dense → Sparse ---
function SparseTensor(t::Tensor{T}) where {T}
    d, m = t.dim, t.level
    s    = t.offsets
    coeffs = Dict{Word,T}()

    len = 1
    @inbounds for k in 0:m
        start = s[k+1] + 1
        if k == 0
            c = t.coeffs[start]
            if !iszero(c); coeffs[Word()] = c; end
        else
            for p in 1:len
                c = t.coeffs[start + p - 1]
                if !iszero(c)
                    rem  = p - 1
                    idxs = Vector{Int}(undef, k)
                    base = (d == 1 ? 1 : div(len, d))
                    @inbounds for j in 1:k
                        q, rem = divrem(rem, base)
                        idxs[j] = q + 1
                        base = (d == 1 ? 1 : div(base, d))
                    end
                    coeffs[Word(idxs)] = c
                end
            end
        end
        len *= d
    end
    return SparseTensor{T}(coeffs, d, m)
end

# --- Sparse → Dense ---
# Explicitly qualify extension of Chen.Tensor to fix warnings
function Tensor(t::SparseTensor{T}) where {T}
    d, m = t.dim, t.level
    out  = Tensor{T}(d, m)
    fill!(out.coeffs, zero(T))
    s = out.offsets

    @inbounds for (w, c) in t.coeffs
        k = length(w)
        if k == 0
            out.coeffs[s[1] + 1] = c
        else
            posm1 = 0
            @inbounds for j in 1:k
                posm1 = posm1 * d + (w.indices[j] - 1)
            end
            idx = s[k + 1] + posm1 + 1
            out.coeffs[idx] = c
        end
    end
    return out
end