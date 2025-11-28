using BenchmarkTools: @belapsed
using ChenSignatures: sig, sig_enzyme, sig_enzyme_horner

println("Comparing signature implementations")
println("="^60)

# Test parameters
N, d, m = 1000, 5, 7
path = randn(N, d)

println("Parameters: N=$N, d=$d, m=$m")
println("="^60)

# 1. sig (optimized, SVector-based)
t_sig = @belapsed sig($path, $m)
println("sig (optimized):           $(round(t_sig * 1000, digits=1)) ms")

# 2. sig_enzyme (simple, Vector-based)
t_enzyme_simple = @belapsed sig_enzyme($path, $m)
println("sig_enzyme (simple):       $(round(t_enzyme_simple * 1000, digits=1)) ms  ($(round(t_enzyme_simple/t_sig, digits=2))x)")

# 3. sig_enzyme_horner (Horner scheme, Vector-based)
t_enzyme_horner = @belapsed sig_enzyme_horner($path, $m)
println("sig_enzyme_horner:         $(round(t_enzyme_horner * 1000, digits=1)) ms  ($(round(t_enzyme_horner/t_sig, digits=2))x)")

println("\n" * "="^60)
println("Speedup comparisons:")
println("  Horner vs Simple: $(round(t_enzyme_simple/t_enzyme_horner, digits=2))x faster")
println("  Horner vs Optimized: $(round(t_enzyme_horner/t_sig, digits=2))x slower")

# Verify correctness
println("\n" * "="^60)
println("Correctness check:")
s1 = sig(path, m)
s2 = sig_enzyme(path, m)
s3 = sig_enzyme_horner(path, m)

println("  sig ≈ sig_enzyme:        ", isapprox(s1, s2, rtol=1e-10))
println("  sig ≈ sig_enzyme_horner: ", isapprox(s1, s3, rtol=1e-10))
println("  Max difference (simple):  ", maximum(abs.(s1 .- s2)))
println("  Max difference (horner):  ", maximum(abs.(s1 .- s3)))

# Test with different sizes
println("\n" * "="^60)
println("Scaling test:")
println("-"^60)
for (test_N, test_d, test_m) in [(100, 3, 4), (500, 4, 5), (1000, 5, 7)]
    test_path = randn(test_N, test_d)
    t1 = @belapsed sig($test_path, $test_m)
    t2 = @belapsed sig_enzyme($test_path, $test_m)
    t3 = @belapsed sig_enzyme_horner($test_path, $test_m)
    
    println("N=$test_N, d=$test_d, m=$test_m:")
    println("  optimized: $(round(t1*1000, digits=1)) ms")
    println("  simple:    $(round(t2*1000, digits=1)) ms  ($(round(t2/t1, digits=2))x)")
    println("  horner:    $(round(t3*1000, digits=1)) ms  ($(round(t3/t1, digits=2))x)")
    println("  horner speedup: $(round(t2/t3, digits=2))x vs simple")
    println()
end