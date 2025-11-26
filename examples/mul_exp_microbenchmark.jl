using Revise
using StaticArrays
using BenchmarkTools
using Random
using Chen

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
const D    = 5      # path dimension
const M    = 5      # signature level
const Nseg = 1000   # number of segments in the path
const N    = Nseg + 1

# Concrete Chen backend with fixed level M
const AT = Chen.Tensor{Float64, M}

rng = MersenneTwister(1234)

# Random path of SVectors
path = [@SVector rand(rng, Float64, D) for _ in 1:N]

# Base tensors / buffers
out = AT(D)
a   = similar(out)
b   = similar(out)
seg = similar(out)

# One example increment
Δ0    = path[2] - path[1]
Δseg  = path[3] - path[2]

# ------------------------------------------------------------------
# exp! microbenchmark
# ------------------------------------------------------------------
println("=== exp! microbenchmark ===")
@btime Chen.exp!($out, $Δ0)

# ------------------------------------------------------------------
# mul! (generic) – arbitrary tensors, not necessarily group-like
# ------------------------------------------------------------------
x1_generic = AT(D)
x2_generic = AT(D)
rand!(x1_generic.coeffs)
rand!(x2_generic.coeffs)

println("\n=== mul! (generic) microbenchmark ===")
@btime Chen.mul!($out, $x1_generic, $x2_generic)

# ------------------------------------------------------------------
# mul_grouplike! – inputs constructed via exp! (so level-0 == 1)
# ------------------------------------------------------------------
x1_gl = AT(D)
x2_gl = AT(D)

Δ1 = @SVector rand(rng, Float64, D)
Δ2 = @SVector rand(rng, Float64, D)

Chen.exp!(x1_gl, Δ1)  # group-like
Chen.exp!(x2_gl, Δ2)  # group-like

println("\n=== mul_grouplike! microbenchmark (group-like inputs) ===")
@btime Chen.mul_grouplike!($out, $x1_gl, $x2_gl)

# ------------------------------------------------------------------
# Per-segment update: exp! + mul!  (a is reinitialised as group-like)
# ------------------------------------------------------------------
println("\n=== per-segment update (exp! + mul!) ===")
@btime begin
    Chen.exp!($seg, $Δseg)          # seg is group-like
    Chen.mul!($b, $a, $seg)
end setup = (Chen.exp!($a, $Δ0))    # a is group-like before each sample

# ------------------------------------------------------------------
# Per-segment update: exp! + mul_grouplike!  (same setup, but specialized mul)
# ------------------------------------------------------------------
println("\n=== per-segment update (exp! + mul_grouplike!) ===")
@btime begin
    Chen.exp!($seg, $Δseg)                # seg is group-like
    Chen.mul_grouplike!($b, $a, $seg)
end setup = (Chen.exp!($a, $Δ0))          # a is group-like before each sample

# ------------------------------------------------------------------
# Full signature_path! for comparison
# ------------------------------------------------------------------
println("\n=== full signature_path! (for comparison) ===")
t_full = @belapsed Chen.signature_path!($out, $path)
println("signature_path!: $(t_full * 1e3) ms total  (Nseg = $Nseg)")
println("per segment ≈ $(t_full / Nseg * 1e6) μs/segment")
