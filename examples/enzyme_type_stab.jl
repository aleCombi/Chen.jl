using ChenSignatures
using Test

# Test with same parameters
N, d, m = 1000, 5, 7
path = randn(N, d)

println("Type stability analysis for sig_enzyme")
println("="^60)

# Check with @code_warntype
println("\n@code_warntype sig_enzyme(path, $m):")
println("-"^60)
@code_warntype sig_enzyme(path, m)

println("\n"^2)
println("="^60)
println("Checking for type instabilities with Test.@inferred:")
println("-"^60)

try
    Test.@inferred sig_enzyme(path, m)
    println("✓ No type instabilities detected!")
catch e
    println("✗ Type instability detected:")
    println(e)
end

println("\n"^2)
println("="^60)
println("Detailed analysis for specific (d, m) combinations:")
println("-"^60)

# Test a few specific cases
for (test_d, test_m) in [(2, 3), (3, 4), (5, 7)]
    test_path = randn(100, test_d)
    println("\nd=$test_d, m=$test_m:")
    try
        Test.@inferred sig_enzyme(test_path, test_m)
        println("  ✓ Type stable")
    catch e
        println("  ✗ Type unstable")
        # Show what the return type inference issue is
        @code_warntype sig_enzyme(test_path, test_m)
    end
end