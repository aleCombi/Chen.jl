using ChenSignatures
using BenchmarkTools
using LinearAlgebra
using Printf
using StaticArrays
using Enzyme

function run_benchmarks()
    println("========================================================================")
    println("       BENCHMARK: Optimized Pointer (sig) vs Enzyme-Safe (sig_enzyme)   ")
    println("========================================================================")
    
    # Test Cases: (N=Length, d=Dimension, m=Level)
    cases = [
        (100,   2, 3),  # Low dim, low level
        (1000,  3, 5),  # Standard usage
        (1000,  5, 5),  # Medium dim
        (500,  10, 3),  # High dim, low level
        (200,   3, 8),  # Low dim, high level (compute bound)
        (10000, 4, 4),  # Long path (memory bound)
    ]

    @printf "%-7s %-3s %-3s | %-10s %-10s | %-8s | %-7s\n" "N" "d" "m" "sig (ms)" "enz (ms)" "Speedup" "Diff"
    println("-"^70)

    for (N, d, m) in cases
        # Generate random path
        path = randn(N, d)

        # 1. Correctness Check
        res_opt = sig(path, m)
        res_enz = sig_enzyme(path, m)
        diff = norm(res_opt - res_enz)
        
        # 2. Benchmark sig (Optimized/Unsafe)
        # We use $ to interpolate variables to avoid global lookup overhead
        t_opt = @belapsed sig($path, $m)
        
        # 3. Benchmark sig_enzyme (Pure Julia/Safe)
        t_enz = @belapsed sig_enzyme($path, $m)

        # Convert to milliseconds
        ms_opt = t_opt * 1000
        ms_enz = t_enz * 1000
        ratio = t_opt / t_enz 
        
        # If ratio < 1.0, sig is faster. If ratio > 1.0, sig_enzyme is faster.
        # "Speedup" here refers to how much faster sig is compared to enz.
        # If sig_enzyme is actually faster, we'll mark it with a *
        
        speedup_str = @sprintf("%.2fx", t_enz / t_opt)
        
        @printf "%-7d %-3d %-3d | %-10.3f %-10.3f | %-8s | %-1.1e\n" N d m ms_opt ms_enz speedup_str diff
    end
    println("-"^70)
    println("Note: Speedup > 1.0x means 'sig' is faster.")
    println()

    # --- ENZYME COMPATIBILITY TEST ---
    println("========================================================================")
    println("                     ENZYME DIFFERENTIATION CHECK                       ")
    println("========================================================================")
    
    N_ad, d_ad, m_ad = 10, 3, 3
    path_test = randn(N_ad, d_ad)
    path_shadow = zeros(N_ad, d_ad) # To store gradient
    
    println("Testing autodiff(Reverse, sig_enzyme, ...)")
    
    function loss_wrapper(p)
        # Simple scalar loss: sum of the signature terms
        s = sig_enzyme(p, m_ad)
        return sum(s)
    end

    try
        # Run AD
        autodiff(Reverse, loss_wrapper, Duplicated(path_test, path_shadow))
        
        grad_norm = norm(path_shadow)
        println("✅ Enzyme Autodiff Successful!")
        println("   Input shape: $(size(path_test))")
        println("   Grad shape:  $(size(path_shadow))")
        println("   Grad Norm:   $(grad_norm)")
        
        if grad_norm == 0.0
            @warn "Gradient is zero. This might indicate broken connectivity in the graph."
        end
    catch e
        println("❌ Enzyme Autodiff Failed")
        showerror(stdout, e)
    end
    println("========================================================================")
end

# Run everything
run_benchmarks()