using ChenSignatures
using Enzyme
using LinearAlgebra

println("Testing return flattened tensor...")

# Test path
path = [0.0 0.0; 1.0 1.0; 2.0 2.0]

# Return flattened tensor (no sum)
function signature_enzyme_inline(path_matrix::Matrix{Float64}, m::Int)
    D = size(path_matrix, 2)
    M = m
    N = size(path_matrix, 1)
    Δ = Vector{Float64}(undef, D)
    
    a = ChenSignatures.Tensor{Float64,D,M}()
    b = ChenSignatures.Tensor{Float64,D,M}()
    seg = ChenSignatures.Tensor{Float64,D,M}()
    
    @inbounds for j in 1:D
        Δ[j] = path_matrix[2, j] - path_matrix[1, j]
    end
    ChenSignatures.non_generated_exp_vec!(a, Δ)
    
    @inbounds for i in 2:N-1
        for j in 1:D
            Δ[j] = path_matrix[i+1, j] - path_matrix[i, j]
        end
        
        ChenSignatures.non_generated_exp_vec!(seg, Δ)
        ChenSignatures.non_generated_mul!(b, a, seg)
        
        a, b = b, a
    end
    
    return ChenSignatures._flatten_tensor(a)  # Return vector, not scalar
end

# Loss takes sum to make scalar for differentiation
loss_enzyme(p) = sum(signature_enzyme_inline(p, 3))

println("\n=== Forward pass ===")
result = signature_enzyme_inline(path, 3)
println("Result length: ", length(result))
println("Sum: ", sum(result))

println("\n=== Enzyme gradient ===")
grad_enzyme = zeros(size(path))
autodiff(Reverse, loss_enzyme, Active, Duplicated(path, grad_enzyme))
display(grad_enzyme)

println("\n\n=== Finite Differences ===")
eps = 1e-6
grad_fd = zeros(size(path))
for i in 1:size(path, 1), j in 1:size(path, 2)
    p_plus = copy(path)
    p_plus[i, j] += eps
    p_minus = copy(path)
    p_minus[i, j] -= eps
    grad_fd[i, j] = (loss_enzyme(p_plus) - loss_enzyme(p_minus)) / (2 * eps)
end
display(grad_fd)

println("\n\n=== Comparison ===")
println("Match: ", isapprox(grad_enzyme, grad_fd, rtol=1e-4))