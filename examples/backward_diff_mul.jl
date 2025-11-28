# test/test_enzyme_mul.jl

using ChenSignatures
using Enzyme
using FiniteDifferences
using Random  # ← Add this!

println("=" ^ 70)
println("ENZYME: Test non_generated_mul!")
println("=" ^ 70)

# ============================================================================
# Test 1: Correctness (non_generated_mul! vs mul!)
# ============================================================================
function test_correctness()
    println("\nTest 1: Correctness check")
    println("-" ^ 60)
    
    D, M = 3, 3
    
    x1 = ChenSignatures.Tensor{Float64, D, M}()
    x2 = ChenSignatures.Tensor{Float64, D, M}()
    
    # Fill with random values
    Random.randn!(x1.coeffs)
    Random.randn!(x2.coeffs)
    
    out_gen = ChenSignatures.Tensor{Float64, D, M}()
    out_non = ChenSignatures.Tensor{Float64, D, M}()
    
    # Compare
    ChenSignatures.mul!(out_gen, x1, x2)
    ChenSignatures.non_generated_mul!(out_non, x1, x2)
    
    err = maximum(abs.(out_gen.coeffs .- out_non.coeffs))
    
    println("Max difference: $err")
    
    if err < 1e-14
        println("✅ PASS - Results match")
        return true
    else
        println("❌ FAIL - Results differ")
        return false
    end
end

# ============================================================================
# Test 2: Enzyme differentiation
# ============================================================================
function test_enzyme()
    println("\nTest 2: Enzyme differentiation")
    println("-" ^ 60)
    
    function loss(x_vec::Vector{Float64})
        D, M = 3, 3
        
        # Create two tensors
        x1 = ChenSignatures.Tensor{Float64, D, M}()
        x2 = ChenSignatures.Tensor{Float64, D, M}()
        out = ChenSignatures.Tensor{Float64, D, M}()
        
        # x1 = exp(x_vec)
        ChenSignatures.non_generated_exp!(x1, x_vec)
        
        # x2 = exp([0.1, 0.1, 0.1])
        ChenSignatures.non_generated_exp!(x2, [0.1, 0.1, 0.1])
        
        # out = x1 ⊗ x2
        ChenSignatures.non_generated_mul!(out, x1, x2)
        
        return sum(out.coeffs)
    end
    
    x = [0.1, 0.2, 0.15]
    grad = zeros(3)
    
    println("Forward: $(loss(x))")
    println("Computing gradient...")
    
    try
        Enzyme.autodiff(Reverse, loss, Active, Duplicated(x, grad))
        
        println("✅ Enzyme: $grad")
        
        # Verify
        fdm = central_fdm(5, 1)
        grad_fd = FiniteDifferences.grad(fdm, loss, x)[1]
        
        println("✅ FinDiff: $grad_fd")
        
        err = maximum(abs.(grad .- grad_fd))
        rel_err = err / maximum(abs.(grad_fd))
        
        println("\nError: $err (relative: $rel_err)")
        
        if rel_err < 1e-5
            println("\n🎉 non_generated_mul! works with Enzyme!")
            return true
        else
            println("\n⚠️ Gradients differ")
            return false
        end
        
    catch e
        println("❌ FAILED")
        showerror(stdout, e, catch_backtrace())
        return false
    end
end

# RUN
test_correctness()
test_enzyme()