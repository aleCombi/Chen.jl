using Chen, ForwardDiff, StaticArrays

function test_signature_ad()
    println("Testing signature_path with ForwardDiff...")
    
    # Simple test: differentiate signature w.r.t. path endpoint
    function f(params)
        # Create a simple 2-segment path
        path = [
            SA[0.0, 0.0],
            SA[1.0, 0.0],
            SA[params[1], params[2]]  # Endpoint depends on params
        ]
        
        sig = signature_path(Chen.Tensor{eltype(params)}, path, 3)
        return sum(sig.coeffs)  # Scalar output for gradient
    end
    
    try
        params = [1.0, 1.0]
        val = f(params)
        grad = ForwardDiff.gradient(f, params)
        
        println("✓ signature_path works with ForwardDiff!")
        println("  Value: $val")
        println("  Gradient w.r.t endpoint: $grad")
        
        # Sanity checks
        if any(isnan, grad) || any(isinf, grad)
            println("⚠ Warning: Gradient has NaN or Inf")
            return false
        end
        
        return true
    catch e
        println("✗ signature_path FAILED with ForwardDiff:")
        println("  $e")
        showerror(stdout, e, catch_backtrace())
        return false
    end
end

test_signature_ad()