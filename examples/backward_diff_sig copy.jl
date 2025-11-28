using ChenSignatures
using Enzyme
using LinearAlgebra

println("Testing return flattened tensor...")

# Test path
path = [0.0 0.0; 1.0 1.0; 2.0 2.0]


# Loss takes sum to make scalar for differentiation
loss_enzyme(p) = sum(ChenSignatures.sig_enzyme(p, 3))

println("\n=== Forward pass ===")
result = ChenSignatures.sig_enzyme(path, 3)
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