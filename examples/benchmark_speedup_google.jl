using ChenSignatures
using BenchmarkTools
using StaticArrays
using LinearAlgebra
using LoopVectorization

# ==============================================================================
# 1. BASELINE
# ==============================================================================
function run_baseline!(a, b, seg, path)
    ChenSignatures._zero!(a); ChenSignatures._write_unit!(a)
    @inbounds for i in 1:length(path)-1
        Δ = path[i+1] - path[i]
        ChenSignatures.exp!(seg, Δ)
        ChenSignatures.mul!(b, a, seg)
        a, b = b, a
    end
    return a
end

# ==============================================================================
# 2. OPTIMIZED (Specialized Kernels for D=3)
# ==============================================================================
module Specialized
    using ChenSignatures
    using LoopVectorization
    using StaticArrays

    # Hardcoded offsets for D=3, up to m=5
    # Level k size = 3^k
    # Offsets[k+1] = start of level k
    # 0, 1, 4, 13, 40, 121, 364
    const OFF = (0, 1, 4, 13, 40, 121, 364)

    @inline function exp_d3!(out::ChenSignatures.Tensor{T}, Δ::SVector{3,T}) where T
        # Unrolled exp! for D=3
        c = out.coeffs
        
        # Level 0
        c[1] = one(T)
        
        # Level 1: Δ
        v1, v2, v3 = Δ[1], Δ[2], Δ[3]
        c[2] = v1; c[3] = v2; c[4] = v3
        
        m = out.level
        m < 2 && return
        
        # Level 2..m: Use recurrence term_k = term_{k-1} ⊗ Δ / k
        # We can implement this very efficiently with known D=3
        inv_k = one(T)
        
        @inbounds for k in 2:m
            inv_k /= T(k)
            prev_start = OFF[k] + 1
            curr_start = OFF[k+1] + 1
            prev_len   = 3^(k-1)
            
            # Vectorize the "tensor product with vector"
            # src[i] -> dest[3i, 3i+1, 3i+2]
            @turbo for i in 0:(prev_len-1)
                val = c[prev_start + i]
                base = curr_start + 3*i
                c[base]     = val * v1 * inv_k
                c[base + 1] = val * v2 * inv_k
                c[base + 2] = val * v3 * inv_k
            end
        end
    end

    @inline function mul_d3!(dest::ChenSignatures.Tensor{T}, A::ChenSignatures.Tensor{T}, B::ChenSignatures.Tensor{T}) where T
        # Specialized Convolution for D=3
        # dest = A ⊗ B
        C = dest.coeffs; Ac = A.coeffs; Bc = B.coeffs
        m = dest.level
        
        # 1. Level 0
        C[1] = Ac[1] * Bc[1]
        
        # 2. Levels 1..m
        @inbounds for k in 1:m
            c_start = OFF[k+1] + 1
            len     = 3^k
            
            # Initialize with A_0 * B_k (if A_0=1, just copy B_k)
            # A[1] is level 0 coeff.
            a0 = Ac[1]
            b_start = OFF[k+1] + 1
            
            @turbo for i in 0:len-1
                C[c_start + i] = a0 * Bc[b_start + i]
            end
            
            # Accumulate: sum_{j=1}^{k} A_j ⊗ B_{k-j}
            # We iterate j (level of A)
            for j in 1:k
                # A_j block
                a_start = OFF[j+1] + 1
                a_len   = 3^j
                
                # B_{k-j} block
                rem_level = k - j
                b_start_rem = OFF[rem_level+1] + 1
                b_len_rem   = 3^rem_level # This is the inner block size
                
                # We interpret A_j ⊗ B_{k-j} as:
                # For each elem in A_j, copy whole B block scaled by it.
                
                # Outer loop over A elements
                for idx_a in 0:(a_len-1)
                    val_a = Ac[a_start + idx_a]
                    
                    # Target position in C
                    # C uses Kronecker layout. A is "higher" bits.
                    # Position = idx_a * (size of B) + idx_b
                    row0 = c_start + idx_a * b_len_rem - 1
                    
                    # Inner vectorized loop over B elements
                    @turbo for idx_b in 1:b_len_rem
                        C[row0 + idx_b] += val_a * Bc[b_start_rem + idx_b - 1]
                    end
                end
            end
        end
    end

    function run_specialized!(a, b, seg, path)
        ChenSignatures._zero!(a); ChenSignatures._write_unit!(a)
        
        @inbounds for i in 1:length(path)-1
            Δ = path[i+1] - path[i]
            
            exp_d3!(seg, Δ)
            mul_d3!(b, a, seg)
            
            a, b = b, a
        end
        return a
    end
end

# ==============================================================================
# 3. BENCHMARK
# ==============================================================================
const N = 1000
const d = 3
const m = 5

println("--- BENCHMARK CONFIGURATION ---")
println("Path: $N x $d, Level: $m")

path_sv = [SVector{d, Float64}(randn(d)) for _ in 1:N]
t_1 = ChenSignatures.Tensor{Float64}(d, m)
t_2 = ChenSignatures.Tensor{Float64}(d, m)
t_3 = ChenSignatures.Tensor{Float64}(d, m)

println("\n1. Correctness")
r1 = run_baseline!(t_1, t_2, t_3, path_sv)
# Reset tensors
t_1c = ChenSignatures.Tensor{Float64}(d, m)
t_2c = ChenSignatures.Tensor{Float64}(d, m)
t_3c = ChenSignatures.Tensor{Float64}(d, m)
r2 = Specialized.run_specialized!(t_1c, t_2c, t_3c, path_sv)

println("Diff: $(norm(r1.coeffs - r2.coeffs))")

println("\n2. Baseline (Generic)")
b_base = @benchmark run_baseline!($t_1, $t_2, $t_3, $path_sv)
display(b_base)

println("\n3. Specialized D=3 (Optimized)")
b_spec = @benchmark Specialized.run_specialized!($t_1c, $t_2c, $t_3c, $path_sv)
display(b_spec)

t_b = median(b_base).time
t_s = median(b_spec).time
println("\nSpeedup: $(round(t_b / t_s, digits=2))x")