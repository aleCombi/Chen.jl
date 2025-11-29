using ChenSignatures
using Zygote
using BenchmarkTools

function benchmark_gradient(N=500, d=5, m=4)
    println("="^70)
    println("JULIA GRADIENT BENCHMARK: N=$N, d=$d, m=$m")
    println("="^70)
    
    # Test data
    path = randn(N, d)
    
    # Forward only
    println("\nForward only:")
    @btime sig($path, $m)
    
    # Forward + Backward
    println("\nForward + Backward (Zygote):")
    grad_fn = x -> sum(sig(x, m))
    @btime gradient($grad_fn, $path)
    
    println("="^70)
end

# Run benchmark
benchmark_gradient(500, 5, 4)