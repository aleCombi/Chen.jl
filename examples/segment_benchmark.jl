using Revise, BenchmarkTools, StaticArrays
using Chen

# length of the flattened tensor series up to level m (assumes d ≥ 2)
_terms(d, m) = div(d^(m + 1) - d, d - 1)

# Build inputs for the Vector endpoint
function make_inputs_vec(d::Int, m::Int, T=Float64)
    @assert d ≥ 2
    a = rand(T, d); b = rand(T, d)
    out = similar(a, _terms(d, m))
    buffer = similar(a, d)
    inv_level = [inv(T(k)) for k in 1:m]          # matches your call site
    return (; a, b, out, buffer, inv_level, m, d, T)
end

# Build inputs for the SVector endpoint
function make_inputs_svec(d::Int, m::Int, T=Float64)
    @assert d ≥ 2
    a = @SVector rand(T, d); b = @SVector rand(T, d)
    out = Vector{T}(undef, _terms(d, m))
    buffer = Vector{T}(undef, d)
    inv_level = [inv(T(k)) for k in 1:m]
    return (; a, b, out, buffer, inv_level, m, d, T)
end

# Optional: compute the ground-truth flattened levels for a straight segment
# using the recurrence  L₁ = Δ,  L_k = (Δ ⊗ L_{k-1}) / k  (lexicographic blocks)
function segment_truth(delta::AbstractVector{T}, m::Int) where {T}
    d = length(delta)
    out = Vector{T}(undef, _terms(d, m))
    idx = 1
    out[idx:idx+d-1] .= delta
    prev = copy(delta)
    idx += d
    for k in 2:m
        prev = kron(delta, prev) .* inv(T(k))
        lenk = d^k
        out[idx:idx+lenk-1] .= prev
        idx += lenk
    end
    return out
end

# Correctness check (Vector endpoint)
function check_segment_correctness_vec(d::Int, m::Int; T=Float64, atol=1e-12)
    inp = make_inputs_vec(d, m, T)
    Δ = inp.b .- inp.a
    segment_truth_vec = segment_truth(Δ, m)
    # call your function
    Chen.segment_signature!(inp.out, inp.a, inp.b, m, inp.buffer, inp.inv_level)
    return maximum(abs.(segment_truth_vec .- inp.out)) ≤ atol
end

# Bench: Vector endpoint
function bench_segment_vec(d::Int, m::Int; T=Float64)
    inp = make_inputs_vec(d, m, T)
    @info "Vector endpoint" d m length(inp.out)
    @btime Chen.segment_signature!($((inp.out)), $((inp.a)), $((inp.b)),
                              $m, $((inp.buffer)), $((inp.inv_level)))
    nothing
end

# Bench: SVector endpoint
function bench_segment_svec(d::Int, m::Int; T=Float64)
    inp = make_inputs_svec(d, m, T)
    @info "SVector endpoint" d m length(inp.out)
    @btime Chen.segment_signature!($((inp.out)), $((inp.a)), $((inp.b)),
                              $m, $((inp.buffer)), $((inp.inv_level)))
    nothing
end

# Example runs:
bench_segment_vec(3, 6)
bench_segment_svec(3, 6)
@show check_segment_correctness_vec(3, 6)
