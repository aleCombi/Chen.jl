using BenchmarkTools: @belapsed
using ChenSignatures: sig, sig_enzyme_horner_svec, sig_enzyme_horner_generated
using StaticArrays
using Enzyme
using Test

println("Testing @generated Horner version")
println("="^60)

# Test parameters
N, d, m = 1000, 5, 7
path = randn(N, d)

# Test 1: Enzyme compatibility
println("\nTest 1: Enzyme compatibility")
println("-"^60)
try
    path_test = randn(5, 2)
    path_shadow = zeros(5, 2)
    
    function loss(path_matrix)
        sig = sig_enzyme_horner_generated(path_matrix, 3)
        return sum(sig)
    end
    
    autodiff(Reverse, loss, Duplicated(path_test, path_shadow))
    println("✓ Enzyme autodiff works!")
    println("  Gradient shape: ", size(path_shadow))
catch e
    println("✗ Enzyme autodiff failed:")
    println("  ", e)
    println("\n  @generated version is NOT Enzyme-compatible.")
    exit(1)
end

# Test 2: Correctness
println("\nTest 2: Correctness check")
println("-"^60)
s1 = sig(path, m)
s2 = sig_enzyme_horner_svec(path, m)
s3 = sig_enzyme_horner_generated(path, m)

println("  sig ≈ horner (svec):      ", isapprox(s1, s2, rtol=1e-10))
println("  sig ≈ horner (generated): ", isapprox(s1, s3, rtol=1e-10))
println("  Max diff (svec):          ", maximum(abs.(s1 .- s2)))
println("  Max diff (generated):     ", maximum(abs.(s1 .- s3)))

# Test 3: Performance
println("\nTest 3: Performance comparison")
println("-"^60)
t_opt = @belapsed sig($path, $m)
t_svec = @belapsed sig_enzyme_horner_svec($path, $m)
t_gen = @belapsed sig_enzyme_horner_generated($path, $m)

println("sig (optimized):           $(round(t_opt * 1000, digits=1)) ms")
println("horner (SVector):          $(round(t_svec * 1000, digits=1)) ms  ($(round(t_svec/t_opt, digits=2))x)")
println("horner (@generated):       $(round(t_gen * 1000, digits=1)) ms  ($(round(t_gen/t_opt, digits=2))x)")
println("\nSpeedup from @generated: $(round(t_svec/t_gen, digits=2))x")

# Test 4: Scaling
println("\nTest 4: Scaling test")
println("-"^60)
for (test_N, test_d, test_m) in [(100, 3, 4), (500, 4, 5), (1000, 5, 7)]
    test_path = randn(test_N, test_d)
    t1 = @belapsed sig($test_path, $test_m)
    t2 = @belapsed sig_enzyme_horner_svec($test_path, $test_m)
    t3 = @belapsed sig_enzyme_horner_generated($test_path, $test_m)
    
    println("N=$test_N, d=$test_d, m=$test_m:")
    println("  optimized:      $(round(t1*1000, digits=1)) ms")
    println("  horner (svec):  $(round(t2*1000, digits=1)) ms  ($(round(t2/t1, digits=2))x)")
    println("  horner (gen):   $(round(t3*1000, digits=1)) ms  ($(round(t3/t1, digits=2))x)")
    println()
end