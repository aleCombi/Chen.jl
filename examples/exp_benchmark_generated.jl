# benchmark/compare_exp_simple.jl

using BenchmarkTools
using StaticArrays
using ChenSignatures

# ============================================================================
# CORRECTNESS CHECK
# ============================================================================
function check_correctness()
    println("=" ^ 70)
    println("CORRECTNESS: ChenSignatures.exp! vs ChenSignaturesexp!")
    println("=" ^ 70)
    
    for D in [2, 3, 5], M in [2, 4, 6]
        x = SVector{D, Float64}(randn(D) * 0.1)
        
        out1 = ChenSignatures.Tensor{Float64, D, M}()
        out2 = ChenSignatures.Tensor{Float64, D, M}()
        
        ChenSignatures.non_generated_exp!(out1, x)
        ChenSignaturesexp!(out2, x)
        
        max_diff = maximum(abs.(out1.coeffs .- out2.coeffs))
        status = max_diff < 1e-14 ? "✓" : "✗ FAIL"
        
        println("D=$D, M=$M: diff = $(max_diff)  $status")
    end
    println()
end

# ============================================================================
# PERFORMANCE BENCHMARK
# ============================================================================
function benchmark_both()
    println("=" ^ 70)
    println("PERFORMANCE COMPARISON")
    println("=" ^ 70)
    
    configs = [
        (D=2, M=3),
        (D=3, M=5),
        (D=5, M=7),
        (D=8, M=5),
    ]
    
    for (D, M) in configs
        println("\n" * "━" ^ 70)
        println("D=$D, M=$M")
        println("━" ^ 70)
        
        x = SVector{D, Float64}(randn(D) * 0.1)
        out1 = ChenSignatures.Tensor{Float64, D, M}()
        out2 = ChenSignatures.Tensor{Float64, D, M}()
        
        # Warmup
        ChenSignatures.non_generated_exp!(out1, x)
        ChenSignaturesexp!(out2, x)
        
        # Benchmark
        t1 = @belapsed ChenSignatures.non_generated_exp!($out1, $x)
        t2 = @belapsed ChenSignaturesexp!($out2, $x)
        
        println("exp!:           $(round(t1 * 1e9, digits=1)) ns")
        println("generated_exp!: $(round(t2 * 1e9, digits=1)) ns")
        
        ratio = t1 / t2
        if ratio > 1.0
            println("→ generated is $(round(ratio, digits=2))× faster")
        else
            println("→ exp! is $(round(1/ratio, digits=2))× faster")
        end
    end
    println("\n" ^ 2)
end

# ============================================================================
# DETAILED BENCHMARK (one case)
# ============================================================================
function detailed_benchmark(D=5, M=5)
    println("=" ^ 70)
    println("DETAILED BENCHMARK: D=$D, M=$M")
    println("=" ^ 70)
    
    x = SVector{D, Float64}(randn(D) * 0.1)
    out1 = ChenSignatures.Tensor{Float64, D, M}()
    out2 = ChenSignatures.Tensor{Float64, D, M}()
    
    println("\nChenSignatures.exp!:")
    @btime ChenSignatures.non_generated_exp!($out1, $x)
    
    println("\nChenSignaturesexp!:")
    @btime ChenSignaturesexp!($out2, $x)
end

# ============================================================================
# RUN
# ============================================================================

check_correctness()
benchmark_both()
detailed_benchmark()
