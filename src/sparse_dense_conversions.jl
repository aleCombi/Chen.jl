# --- Dense → Sparse (levels 0..m), respecting padding ---
function SparseTensor(t::Chen.Tensor{T}) where {T}
    d, m = t.dim, t.level
    s    = t.offsets                     # 0-based starts for levels 0..m; length m+2
    coeffs = Dict{Word,T}()

    len = 1                              # = d^k, start at k=0
    @inbounds for k in 0:m
        start = s[k+1] + 1               # 1-based start of level k block
        if k == 0
            c = t.coeffs[start]
            if !iszero(c)
                coeffs[Word()] = c
            end
        else
            # scan exactly d^k entries (skip any padding after the block)
            for p in 1:len
                c = t.coeffs[start + p - 1]
                if !iszero(c)
                    rem  = p - 1
                    idxs = Vector{Int}(undef, k)
                    base = (d == 1 ? 1 : div(len, d))   # = d^(k-1)
                    @inbounds for j in 1:k
                        q, rem = divrem(rem, base)
                        idxs[j] = q + 1
                        base = (d == 1 ? 1 : div(base, d))
                    end
                    coeffs[Word(idxs)] = c
                end
            end
        end
        len *= d                          # advance to d^(k+1)
    end
    return Chen.SparseTensor{T}(coeffs, d, m)
end

# --- Sparse → Dense (levels 0..m) ---
function Tensor(t::Chen.SparseTensor{T}) where {T}
    d, m = t.dim, t.level
    out  = Chen.Tensor{T}(d, m)     # has padded offsets
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
            idx = s[k + 1] + posm1 + 1        # start of level k + position
            out.coeffs[idx] = c
        end
    end
    return out
end