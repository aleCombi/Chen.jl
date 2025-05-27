using Revise, PathSignatures, PythonCall
@py import iisignature
@py import numpy as np
using BenchmarkTools

PathSignatures.signature_words(2,2) |> collect
PathSignatures.all_signature_words(2,2) |> collect

function segment_signature(f, a, b, m)
    displacement = f(b) - f(a)
    d = length(displacement)
    T = eltype(displacement)

    # Total number of signature terms from level 1 to m:
    total_terms = div(d^(m + 1) - d, d - 1)

    sig = Vector{T}(undef, total_terms)
    idx = 1

    # First level
    curlen = d
    current = view(sig, idx:idx+curlen-1)
    current .= displacement
    idx += curlen
    prevlen = curlen

    for level in 2:m
        curlen = d^level
        current = view(sig, idx:idx+curlen-1)
        _segment_level!(current, displacement, level, view(sig, idx - prevlen:idx - 1))
        idx += curlen
        prevlen = curlen
    end

    return sig
end

function _segment_level!(out::AbstractVector{T}, Δ::AbstractVector{T}, m::Int, previous::AbstractVector{T}) where T
    d, n = length(Δ), length(previous)
    scale = inv(T(m))
    @inbounds for i in 1:d
        for j in 1:n
            out[(i - 1) * n + j] = scale * Δ[i] * previous[j]
        end
    end
end

function chen_product!(out::Vector{T}, x1::Vector{T}, x2::Vector{T}, d::Int, m::Int) where T
    level_sizes = [d^k for k in 1:m]
    offsets = cumsum([0; level_sizes])

    for k in 1:m
        out_k = view(out, offsets[k]+1 : offsets[k+1])
        fill!(out_k, 0)

        for i in 0:k
            a = i == 0     ? nothing : view(x1, offsets[i]+1:offsets[i+1])
            b = (k - i) == 0 ? nothing : view(x2, offsets[k - i]+1:offsets[k - i + 1])
            na = i == 0     ? 1 : length(a)
            nb = (k - i) == 0 ? 1 : length(b)

            @inbounds @simd for ai in 1:na
                a_val = i == 0 ? one(T) : a[ai]
                @simd for bi in 1:nb
                    b_val = (k - i) == 0 ? one(T) : b[bi]
                    out_k[(ai - 1) * nb + bi] += a_val * b_val
                end
            end
        end
    end
    return out
end

function signature_path(f, ts::AbstractVector{<:Real}, m::Int)
    d = length(f(ts[1]))
    T = eltype(f(ts[1]))

    # Number of signature terms (excluding level 0)
    total_terms = div(d^(m + 1) - d, d - 1)

    # Compute signature of first segment
    sig = segment_signature(f, ts[1], ts[2], m)

    # Use a scratch buffer for chen_product!
    buf = Vector{T}(undef, total_terms)

    for i in 2:length(ts)-1
        sig_next = segment_signature(f, ts[i], ts[i+1], m)
        chen_product!(buf, sig, sig_next, d, m)
        sig, buf = buf, sig  # swap buffers
    end

    return sig
end

function segment_signature_three_points(f, a, b, c, m)
    d = length(f(0.0))

    x_ab = segment_signature(f, a, b, m)
    x_bc = segment_signature(f, b, c, m)

    total_terms = div(d^(m + 1) - d, d - 1)
    x_ac = Vector{eltype(x_ab)}(undef, total_terms)

    chen_product!(x_ac, x_ab, x_bc, d, m)

    return x_ac
end


# f(t) = [t, 2t]
# a, b = 0.0, 1.0
# m = 20

# sig = segment_signature(f, a, b, m)
# # @show sig
# @show length(sig)  # should be 1 + 2 + 4 + 8 = 15 for d = 2, m = 3

# x0 = f(a)
# x1 = f(b)

# path = vcat(x0', x1')  # shape (2, d), one row per point
# path_np = np.asarray(path; order="C")

# sig_py = iisignature.sig(path_np, m)

# @btime segment_signature($f, $a, $b, $m)
# @btime $iisignature.sig($path_np, $m)


####

# Define path
f(t) = [t, 2t]
ts = range(0, stop=1, length=10)  # 9 segments, 10 points
m = 10
d = length(f(0.0))

# Julia: N-point signature via Chen
sig_julia = signature_path(f, ts, m)

# Python: full path signature
path = reduce(vcat, [f(t)' for t in ts])  # Matrix (10, d) flattened row-wise
path_np = np.asarray(path; order="C")
sig_py = iisignature.sig(path_np, m)
sig_py_julia = pyconvert(Vector{Float64}, sig_py)

# ✅ Validate match
@assert isapprox(sig_julia, sig_py_julia; atol=1e-12)

# 🕒 Benchmark
println("Benchmarking:")
@btime signature_path($f, $ts, $m)
@btime $iisignature.sig($path_np, $m)