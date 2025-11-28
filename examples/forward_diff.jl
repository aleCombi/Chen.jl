using ForwardDiff
using LinearAlgebra
using ChenSignatures  # assuming sig is exported from here

# Problem size (keep small for FD comparison)
N, d, m = 4, 2, 3

# Base path
path0 = randn(N, d)

# Flattened interface for ForwardDiff & FD
function sig_flattened(x::AbstractVector)
    P = reshape(x, N, d)
    s = sig(P, m)          # your signature function
    return s               # should be a Vector
end

x0 = vec(path0)  # length = N*d

# 1. ForwardDiff Jacobian
J_ad = ForwardDiff.jacobian(sig_flattened, x0)
println("AD Jacobian size: ", size(J_ad))

# 2. Finite-difference Jacobian (central differences)
function fd_jacobian(f, x; eps = 1e-6)
    y0 = f(x)
    n = length(x)
    m = length(y0)
    J = zeros(eltype(y0), m, n)

    for j in 1:n
        e = zeros(eltype(x), n)
        e[j] = eps
        fp = f(x .+ e)
        fm = f(x .- e)
        J[:, j] = (fp .- fm) ./ (2eps)
    end

    return J
end

J_fd = fd_jacobian(sig_flattened, x0; eps = 1e-6)
println("FD Jacobian size: ", size(J_fd))

# 3. Compare
diff = J_ad .- J_fd
max_abs_err = maximum(abs, diff)
denom = max(1.0, maximum(abs, J_ad))
rel_err = max_abs_err / denom

println("Max abs difference: ", max_abs_err)
println("Relative max error: ", rel_err)
