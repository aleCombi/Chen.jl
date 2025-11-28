using ChenSignatures
using Enzyme
using Test

println("Testing Enzyme compatibility of update_signature_horner_enzyme!")
println("="^60)

# Simple test case
D = 2
M = 3
N = 5

path = randn(N, D)
z = Vector{Float64}(undef, D)

# Test 1: Check if the function itself works
println("\nTest 1: Basic functionality")
println("-"^60)
try
    a = Tensor{Float64, D, M}()
    max_buffer_size = D^(M-1)
    B1 = Vector{Float64}(undef, max_buffer_size)
    B2 = Vector{Float64}(undef, max_buffer_size)
    
    for i in 1:N-1
        for j in 1:D
            z[j] = path[i+1, j] - path[i, j]
        end
        ChenSignatures.update_signature_horner_enzyme!(a, z, B1, B2)
    end
    
    println("✓ Function executes successfully")
    println("  Result tensor has $(length(a.coeffs)) coefficients")
catch e
    println("✗ Function failed:")
    println("  ", e)
end

# Test 2: Check Enzyme on the update function
println("\nTest 2: Enzyme autodiff on update_signature_horner_enzyme!")
println("-"^60)
try
    a = Tensor{Float64, D, M}()
    a_shadow = Tensor{Float64, D, M}()
    max_buffer_size = D^(M-1)
    B1 = Vector{Float64}(undef, max_buffer_size)
    B2 = Vector{Float64}(undef, max_buffer_size)
    B1_shadow = zeros(Float64, max_buffer_size)
    B2_shadow = zeros(Float64, max_buffer_size)
    
    z_test = [1.0, 2.0]
    z_shadow = zeros(D)
    
    autodiff(Reverse, ChenSignatures.update_signature_horner_enzyme!,
             Duplicated(a, a_shadow),
             Duplicated(z_test, z_shadow),
             Duplicated(B1, B1_shadow),
             Duplicated(B2, B2_shadow))
    
    println("✓ Enzyme autodiff works!")
    println("  z_shadow = ", z_shadow)
catch e
    println("✗ Enzyme autodiff failed:")
    println("  ", e)
    if isa(e, ErrorException) && occursin("Enzyme", string(e))
        println("\n  This might be an Enzyme compatibility issue")
    end
end

# Test 3: Check Enzyme on the full sig_enzyme_horner function
println("\nTest 3: Enzyme autodiff on full sig_enzyme_horner")
println("-"^60)
try
    path_test = randn(5, 2)
    path_shadow = zeros(5, 2)
    
    function loss(path_matrix)
        sig = ChenSignatures.sig_enzyme_horner(path_matrix, 3)
        return sum(sig)
    end
    
    autodiff(Reverse, loss, Duplicated(path_test, path_shadow))
    
    println("✓ Full function Enzyme autodiff works!")
    println("  Gradient shape: ", size(path_shadow))
    println("  Gradient sample: ", path_shadow[1:2, :])
catch e
    println("✗ Full function Enzyme autodiff failed:")
    println("  ", e)
end

println("\n" * "="^60)
println("Summary:")
println("  If all tests pass, the Horner version is Enzyme-compatible!")