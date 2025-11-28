# Complete working example with signature_from_matrix
using ChenSignatures
using Enzyme
using FiniteDifferences

println("=" ^ 70)
println("COMPLETE TEST: signature_from_matrix with Enzyme")
println("=" ^ 70)

D, M = 3, 3

# Pre-allocate ALL buffers
out = ChenSignatures.Tensor{Float64, D, M}()
seg = ChenSignatures.Tensor{Float64, D, M}()
tmp = ChenSignatures.Tensor{Float64, D, M}()
Δ_vec = Vector{Float64}(undef, D)

# Shadows for Enzyme
out_shadow = ChenSignatures.Tensor{Float64, D, M}()
seg_shadow = ChenSignatures.Tensor{Float64, D, M}()
tmp_shadow = ChenSignatures.Tensor{Float64, D, M}()
Δ_vec_shadow = zeros(Float64, D)

# Test matrix
path_matrix = [
    0.0  0.0  0.0;
    0.3  0.2  0.1;
    0.6  0.5  0.4;
    1.0  0.8  0.7
]

# Loss function: all buffers passed as arguments (not captured)
function loss_with_buffers(
    out::ChenSignatures.Tensor{Float64,3,3},
    seg::ChenSignatures.Tensor{Float64,3,3},
    tmp::ChenSignatures.Tensor{Float64,3,3},
    Δ_vec::Vector{Float64},
    path_matrix::Matrix{Float64},
    endpoint::Vector{Float64}
)
    # Mutate path matrix
    path_matrix[4, 1] = endpoint[1]
    path_matrix[4, 2] = endpoint[2]
    path_matrix[4, 3] = endpoint[3]
    
    # Compute signature using the mutating function
    N = size(path_matrix, 1)
    
    # Initialize out
    fill!(out.coeffs, 0.0)
    out.coeffs[out.offsets[1] + 1] = 1.0
    
    # First segment
    @inbounds for j in 1:D
        Δ_vec[j] = path_matrix[2, j] - path_matrix[1, j]
    end
    ChenSignatures.non_generated_exp!(seg, Δ_vec)
    
    # Copy
    @inbounds for i in eachindex(out.coeffs, seg.coeffs)
        out.coeffs[i] = seg.coeffs[i]
    end
    
    # Remaining segments
    @inbounds for i in 2:N-1
        for j in 1:D
            Δ_vec[j] = path_matrix[i+1, j] - path_matrix[i, j]
        end
        
        ChenSignatures.non_generated_exp!(seg, Δ_vec)
        ChenSignatures.non_generated_mul!(tmp, out, seg)
        
        for k in eachindex(out.coeffs, tmp.coeffs)
            out.coeffs[k] = tmp.coeffs[k]
        end
    end
    
    return sum(out.coeffs)
end

# Test forward pass
endpoint = [1.0, 0.8, 0.7]
result = loss_with_buffers(out, seg, tmp, Δ_vec, path_matrix, endpoint)
println("Forward pass: $result")

# Test with Enzyme
endpoint_test = [1.0, 0.8, 0.7]
grad = zeros(3)

println("\nComputing gradient with Enzyme...")
try
    Enzyme.autodiff(
        Reverse,
        loss_with_buffers,
        Active,
        Duplicated(out, out_shadow),
        Duplicated(seg, seg_shadow),
        Duplicated(tmp, tmp_shadow),
        Duplicated(Δ_vec, Δ_vec_shadow),
        Const(path_matrix),  # Path matrix is Const
        Duplicated(endpoint_test, grad)
    )
    
    println("✅ Enzyme SUCCESS!")
    println("   Gradient: $grad")
    
    # Verify with finite differences
    println("\nVerifying with FiniteDifferences...")
    
    function loss_for_fd(ep)
        loss_with_buffers(out, seg, tmp, Δ_vec, path_matrix, ep)
    end
    
    fdm = central_fdm(5, 1)
    grad_fd = FiniteDifferences.grad(fdm, loss_for_fd, endpoint)[1]
    
    println("   FiniteDiff: $grad_fd")
    
    err = maximum(abs.(grad .- grad_fd))
    rel_err = err / maximum(abs.(grad_fd))
    
    println("\n   Abs Error: $err")
    println("   Rel Error: $rel_err")
    
    if rel_err < 1e-5
        println("\n🎉 PERFECT! Gradients match!")
    else
        println("\n⚠️ Gradients differ")
    end
    
catch e
    println("❌ FAILED")
    showerror(stdout, e, catch_backtrace())
end