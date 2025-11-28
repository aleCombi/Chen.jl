using BenchmarkTools
using ChenSignatures

# Test with same parameters as Python
N, d, m = 1000, 5, 7
path = randn(N, d)

println("Benchmarking: N=$N, d=$d, m=$m")
println("="^60)

# Benchmark sig (optimized)
t_sig = @belapsed sig($path, $m)
println("sig (optimized):     $(round(t_sig * 1000, digits=1)) ms")

# Benchmark sig_enzyme
t_enzyme = @belapsed sig_enzyme($path, $m)
println("sig_enzyme:          $(round(t_enzyme * 1000, digits=1)) ms")

println("\nRatio: sig_enzyme is $(round(t_sig/t_enzyme, digits=2))x relative to sig")

# Verify they give same results
s1 = sig(path, m)
s2 = sig_enzyme(path, m)
println("\nResults match: ", isapprox(s1, s2, rtol=1e-10))