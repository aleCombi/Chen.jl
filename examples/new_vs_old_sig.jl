using ChenSignatures
using Enzyme
using BenchmarkTools
using Test
using LinearAlgebra
using Printf
using Random

# ==============================================================================
# 0. Definitions & Helpers
# ==============================================================================

# Map the optimized implementation
const sig_new = ChenSignatures.sig_on_the_fly

# A scalar loss function for testing gradients: L = 0.5 * ||sig(path)||^2
function scalar_loss(path::Matrix{Float64}, m::Int)
    s = sig_new(path, m)
    # simple sum of squares to reduce to scalar
    return 0.5 * sum(abs2, s)
end

# Structure to hold benchmark results
struct BenchResult
    N::Int
    D::Int
    m::Int
    t_old_fwd::Float64   # Time (seconds)
    t_new_fwd::Float64
    t_new_bwd::Float64
    alloc_old::Int       # Allocations (bytes)
    alloc_new::Int
end

# Pretty printer for the results
function print_results_table(results::Vector{BenchResult})
    println("\n" * "="^110)
    @printf("%-7s %-4s %-4s | %-12s %-12s %-10s | %-12s %-12s | %-12s\n", 
        "N", "D", "m", "Old(Fwd)", "New(Fwd)", "Speedup", "AllocOld", "AllocNew", "New(Bwd)")
    println("-"^110)
    
    for r in results
        speedup = r.t_old_fwd / r.t_new_fwd
        
        # Format times (ms or μs)
        t_fmt(t) = t < 1e-3 ? @sprintf("%.1f μs", t*1e6) : @sprintf("%.2f ms", t*1e3)
        
        # Format memory
        mem_fmt(b) = b > 1e6 ? @sprintf("%.1f MiB", b/1024^2) : @sprintf("%.1f KiB", b/1024)

        @printf("%-7d %-4d %-4d | %-12s %-12s %-10.2fx | %-12s %-12s | %-12s\n",
            r.N, r.D, r.m,
            t_fmt(r.t_old_fwd), t_fmt(r.t_new_fwd), speedup,
            mem_fmt(r.alloc_old), mem_fmt(r.alloc_new),
            t_fmt(r.t_new_bwd)
        )
    end
    println("="^110 * "\n")
end

# ==============================================================================
# 1. Correctness "Sanity Check"
# ==============================================================================
println("Running Correctness Sanity Check...")
let
    path_check = rand(20, 3)
    m_check = 4
    res_old = sig(path_check, m_check)
    res_new = sig_new(path_check, m_check)
    
    # Check values match
    if res_old ≈ res_new
        println("✅ Correctness Passed: sig and sig_new outputs match.")
    else
        error("❌ Correctness Failed: Outputs do not match!")
    end

    # Check Enzyme Gradient runs without crashing on sig_new
    dx = zero(path_check)
    try
        Enzyme.autodiff(Reverse, scalar_loss, Active, Duplicated(path_check, dx), Const(m_check))
        println("✅ Differentiability Passed: Enzyme ran successfully on sig_new.")
    catch e
        println("❌ Differentiability Failed on sig_new: ", e)
    end
end

# ==============================================================================
# 2. Generalized Benchmark Loop
# ==============================================================================

# Define the scenarios to test
# Be careful: Signature size grows as D^m. 
# D=5, m=6 is 15,625 elements per path (manageable).
scenarios = [
    # (N, D, m)
    (100,  2, 3),  # Small, Low Dim
    (1000, 2, 4),  # Long path
    (100,  3, 4),  # Standard Use Case
    (500,  3, 5),  # Heavier
    (100,  5, 3),  # High Dimension, Low Level
    (50,   10, 2), # Very High Dim
]

results = BenchResult[]

println("\nStarting Benchmarks (Warmup included in @benchmark)...")

for (N, D, m) in scenarios
    print("Testing N=$N, D=$D, m=$m ... ")
    
    path = rand(Float64, N, D)
    d_path = zero(path) # Shadow memory for gradient

    # 1. Benchmark Old Forward
    b_old = @benchmark sig($path, $m) samples=50 seconds=2
    
    # 2. Benchmark New Forward
    b_new = @benchmark sig_new($path, $m) samples=100 seconds=2
    
    # 3. Benchmark New Backward (Enzyme)
    # Note: We duplicate `path` to tell Enzyme we want gradients w.r.t it.
    b_bwd = @benchmark Enzyme.autodiff(Reverse, scalar_loss, Active, Duplicated($path, $d_path), Const($m)) samples=50 seconds=2

    # Collect stats (using median for robustness)
    push!(results, BenchResult(
        N, D, m,
        median(b_old).time / 1e9,
        median(b_new).time / 1e9,
        median(b_bwd).time / 1e9,
        memory(median(b_old)),
        memory(median(b_new))
    ))
    println("Done.")
end

# ==============================================================================
# 3. Final Report
# ==============================================================================

print_results_table(results)

println("Interpretation:")
println("1. Speedup: Expect 2x - 100x depending on N and allocation overhead.")
println("2. Allocations: sig_new should have significantly fewer allocations (avoids Vector{SVector}).")
println("3. Backward: This column validates that differentiation is performant.")