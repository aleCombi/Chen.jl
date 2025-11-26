using ChenSignatures
using BenchmarkTools
using StaticArrays
using LoopVectorization

const FACTORIAL_RECIP = Float64[
    1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664,
    0.008333333333333333, 0.001388888888888889, 0.0001984126984126984,
]

function fused_update_optimized!(
    out::ChenSignatures.Tensor{T},
    accum::ChenSignatures.Tensor{T},
    Δ::AbstractVector{T}
) where {T}
    
    d = out.dim
    m = out.level
    offsets = out.offsets
    
    # Precompute Δ powers
    delta_powers = Vector{Vector{T}}(undef, m)
    delta_powers[1] = Vector{T}(undef, d)
    @inbounds @simd for i in 1:d
        delta_powers[1][i] = Δ[i]
    end
    
    for p in 2:m
        len_prev = d^(p-1)
        len_cur = d^p
        delta_powers[p] = Vector{T}(undef, len_cur)
        
        @turbo for i in 1:len_prev, j in 1:d
            idx = (i-1)*d + j
            delta_powers[p][idx] = delta_powers[p-1][i] * Δ[j]
        end
    end
    
    @inbounds begin
        a0 = accum.coeffs[offsets[1] + 1]
        out.coeffs[offsets[1] + 1] = a0
        
        if m >= 1
            out_start = offsets[2] + 1
            accum_start = offsets[2] + 1
            
            @turbo for i in 1:d
                out.coeffs[out_start + i - 1] = a0 * Δ[i] + accum.coeffs[accum_start + i - 1]
            end
        end
        
        for k in 2:m
            out_start = offsets[k+1] + 1
            len_k = d^k
            
            accum_k_start = offsets[k+1] + 1
            @turbo for i in 1:len_k
                out.coeffs[out_start + i - 1] = accum.coeffs[accum_k_start + i - 1]
            end
            
            scale = k <= length(FACTORIAL_RECIP) ? T(FACTORIAL_RECIP[k+1]) : inv(T(factorial(k)))
            @turbo for i in 1:len_k
                out.coeffs[out_start + i - 1] += a0 * scale * delta_powers[k][i]
            end
            
            len_j = d
            for j in 1:(k-1)
                accum_j_start = offsets[j+1] + 1
                power = k - j
                scale = power <= length(FACTORIAL_RECIP) ? 
                        T(FACTORIAL_RECIP[power+1]) : 
                        inv(T(factorial(power)))
                
                len_delta = d^power
                
                @turbo for i in 1:len_j, p in 1:len_delta
                    idx = out_start + (i-1)*len_delta + p - 1
                    out.coeffs[idx] += scale * accum.coeffs[accum_j_start + i - 1] * delta_powers[power][p]
                end
                
                len_j *= d
            end
        end
    end
    
    return out
end

# ============================================================================
# SINGLE TEST CASE - d=5, m=2 (known to work)
# ============================================================================

println("="^70)
println("Fused Kernel Speed Test: d=5, m=2")
println("="^70)

# Setup
d, m = 5, 2
ts = range(0.0, stop=1.0, length=100)
path = [SVector{d,Float64}(ntuple(i -> (i == 1 ? t : 2t), d)) for t in ts]

accum = ChenSignatures.Tensor{Float64}(d, m)
Δ1 = path[2] - path[1]
ChenSignatures.exp!(accum, Δ1)

Δ2 = path[3] - path[2]

# Test correctness
seg = ChenSignatures.Tensor{Float64}(d, m)
out_old = ChenSignatures.Tensor{Float64}(d, m)
out_new = ChenSignatures.Tensor{Float64}(d, m)

ChenSignatures.exp!(seg, Δ2)
ChenSignatures.mul!(out_old, accum, seg)
fused_update_optimized!(out_new, accum, Δ2)

max_diff = maximum(abs.(out_old.coeffs .- out_new.coeffs))
println("\nCorrectness check:")
println("  Max difference: $max_diff")

if max_diff < 1e-10
    println("  ✅ PASS - Results match!\n")
    
    # Benchmark
    println("Benchmarking...")
    println("\n⏱️  Current (exp! + mul!):")
    t1 = @benchmark begin
        ChenSignatures.exp!($seg, $Δ2)
        ChenSignatures.mul!($out_old, $accum, $seg)
    end samples=1000
    display(t1)
    
    println("\n\n⏱️  Fused kernel:")
    t2 = @benchmark fused_update_optimized!($out_new, $accum, $Δ2) samples=1000
    display(t2)
    
    time_old = median(t1.times) / 1e3
    time_new = median(t2.times) / 1e3
    speedup = time_old / time_new
    
    mem_old = t1.memory / 1024
    mem_new = t2.memory / 1024
    
    println("\n" * "="^70)
    println("RESULTS:")
    println("="^70)
    println("  Current: $(round(time_old, digits=2)) μs, $(round(mem_old, digits=1)) KiB")
    println("  Fused:   $(round(time_new, digits=2)) μs, $(round(mem_new, digits=1)) KiB")
    println()
    if speedup > 1.0
        println("  ✅ Speedup: $(round(speedup, digits=2))x FASTER")
    else
        println("  ❌ Slowdown: $(round(1/speedup, digits=2))x SLOWER")
    end
    
    if mem_new < mem_old && mem_new > 0
        println("  Memory saved: $(round(100*(1-mem_new/mem_old), digits=1))%")
    end
    println("="^70)
else
    println("  ❌ FAIL - Results don't match!")
    println("\nFirst 10 coefficients:")
    for i in 1:min(10, length(out_old.coeffs))
        println("  [$i] current=$(out_old.coeffs[i]), fused=$(out_new.coeffs[i])")
    end
end