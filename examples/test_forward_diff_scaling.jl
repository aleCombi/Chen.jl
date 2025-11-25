using Chen, ForwardDiff, StaticArrays, BenchmarkTools

println("Testing ForwardDiff scaling with path length...")
println()

# Test: Does ForwardDiff scale with N or with num_params?
function test_scaling()
    # Fixed: 2 parameters (control points)
    # Variable: path length N
    
    function make_loss(N)
        return function(params)
            # Generate N-point path from 2 control points
            t = range(0, 1, length=N)
            path = [SA[params[1] * tt, params[2] * (1-tt)] for tt in t]
            sig = signature_path(Chen.Tensor{eltype(params)}, path, 3)
            return sum(sig.coeffs)
        end
    end
    
    for N in [10, 100, 1000]
        loss_fn = make_loss(N)
        params = [1.0, 2.0]
        
        # Time forward pass
        t_forward = @belapsed $loss_fn($params) samples=100
        
        # Time gradient (ForwardDiff)
        t_grad = @belapsed ForwardDiff.gradient($loss_fn, $params) samples=100
        
        overhead = t_grad / t_forward
        
        println("N=$N:")
        println("  Forward: $(round(t_forward*1e6, digits=1)) μs")
        println("  Gradient: $(round(t_grad*1e6, digits=1)) μs")
        println("  Overhead: $(round(overhead, digits=1))× (should be ~2-3×, NOT $(N)×)")
        println()
    end
    
    println("Conclusion:")
    println("If overhead stays ~2-3× regardless of N, then ForwardDiff")
    println("scales with NUM_PARAMS (2), not PATH_LENGTH (N).")
end

test_scaling()
