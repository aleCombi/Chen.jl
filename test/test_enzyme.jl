using ChenSignatures
using Test
using Enzyme
using ForwardDiff
using LinearAlgebra
using Random

@testset "sig - Comprehensive Gradient Tests" begin
    # Skip Enzyme tests on Julia 1.12+ where support is experimental
    # See: https://github.com/EnzymeAD/Enzyme.jl/issues/2699
    # The wrapper pattern in signature_path! causes issues with Enzyme's AD on 1.12
    # ChainRules/Zygote AD still works fine via the rrule definition
    if VERSION >= v"1.12"
        @info "Skipping Enzyme tests on Julia $(VERSION) (Enzyme's 1.12+ support is experimental)"
        return
    end
    # Make all tests deterministic
    Random.seed!(0)

    @testset "Basic functionality" begin
        path = [0.0 0.0; 1.0 1.0; 2.0 3.0]

        result = sig(path, 3)
        @test result isa Vector{Float64}
        @test length(result) == 2 + 4 + 8  # d^1 + d^2 + d^3 for d=2, m=3
        @test all(isfinite, result)
    end

    @testset "Edge cases: zero and short paths" begin
        # Zero path
        path0 = zeros(4, 2)
        sig0 = sig(path0, 3)
        @test all(isfinite, sig0)

        # Minimal path (N = 2)
        path2 = randn(2, 2)
        sig2 = sig(path2, 3)
        @test all(isfinite, sig2)
    end

    # Shared FD parameters
    eps = 1e-6
    rtol = 1e-4
    atol = 1e-8

    @testset "Enzyme vs Finite Differences – sum loss" begin
        path = randn(4, 2)
        loss(p) = sum(sig(p, 3))

        grad_enzyme = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad_enzyme))

        grad_fd = zeros(size(path))
        for i in 1:size(path,1), j in 1:size(path,2)
            p_plus  = copy(path);  p_plus[i,j]  += eps
            p_minus = copy(path); p_minus[i,j] -= eps
            grad_fd[i,j] = (loss(p_plus) - loss(p_minus)) / (2eps)
        end

        @test isapprox(grad_enzyme, grad_fd; rtol=rtol, atol=atol)
    end

    @testset "Enzyme vs Finite Differences – specific coefficient" begin
        path = randn(4, 2)
        loss(p) = sig(p, 3)[5]

        grad_enzyme = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad_enzyme))

        grad_fd = zeros(size(path))
        for i in 1:size(path,1), j in 1:size(path,2)
            p_plus  = copy(path);  p_plus[i,j]  += eps
            p_minus = copy(path); p_minus[i,j] -= eps
            grad_fd[i,j] = (loss(p_plus) - loss(p_minus)) / (2eps)
        end

        @test isapprox(grad_enzyme, grad_fd; rtol=rtol, atol=atol)
    end

    @testset "Enzyme vs Finite Differences – L2 loss" begin
        path = randn(4, 2)
        loss(p) = sum(abs2, sig(p, 3))

        grad_enzyme = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad_enzyme))

        grad_fd = zeros(size(path))
        for i in 1:size(path,1), j in 1:size(path,2)
            p_plus  = copy(path);  p_plus[i,j]  += eps
            p_minus = copy(path); p_minus[i,j] -= eps
            grad_fd[i,j] = (loss(p_plus) - loss(p_minus)) / (2eps)
        end

        @test isapprox(grad_enzyme, grad_fd; rtol=rtol, atol=atol)
    end

    @testset "Enzyme gradient is finite on normal input" begin
        path = randn(4, 2)
        loss(p) = sum(sig(p, 3))

        grad = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad))

        @test all(isfinite, grad)
    end

    @testset "3D paths – correctness and gradient spot check" begin
        path3 = randn(4, 3)

        result = sig(path3, 3)
        @test length(result) == 3 + 9 + 27
        @test all(isfinite, result)

        loss(p) = sum(sig(p, 3))

        grad = zeros(size(path3))
        autodiff(Reverse, loss, Active, Duplicated(path3, grad))

        # Spot check
        i, j = 2, 1
        p_plus  = copy(path3); p_plus[i,j]  += eps
        p_minus = copy(path3); p_minus[i,j] -= eps
        fd_ij = (loss(p_plus) - loss(p_minus)) / (2eps)

        @test isapprox(grad[i,j], fd_ij; rtol=rtol, atol=atol)
    end

    @testset "Longer paths – randomized FD spot checks" begin
        path = randn(20, 2)
        loss(p) = sum(sig(p, 3))

        grad = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad))

        for _ in 1:3
            i = rand(1:size(path,1))
            j = rand(1:size(path,2))

            p_plus  = copy(path); p_plus[i,j]  += eps
            p_minus = copy(path); p_minus[i,j] -= eps
            fd_ij = (loss(p_plus) - loss(p_minus)) / (2eps)

            @test isapprox(grad[i,j], fd_ij; rtol=rtol, atol=atol)
        end
    end

    @testset "Enzyme vs ForwardDiff vs FD (small case)" begin
        path = randn(3, 2)
        loss(p) = sum(sig(p, 3))

        # Enzyme
        grad_enzyme = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad_enzyme))

        # Finite Differences
        grad_fd = similar(grad_enzyme)
        for i in 1:size(path,1), j in 1:size(path,2)
            p_plus  = copy(path); p_plus[i,j]  += eps
            p_minus = copy(path); p_minus[i,j] -= eps
            grad_fd[i,j] = (loss(p_plus) - loss(p_minus)) / (2eps)
        end

        # ForwardDiff (requires flattening)
        loss_flat(x) = loss(reshape(x, size(path)))
        grad_fwd_flat = ForwardDiff.gradient(loss_flat, vec(path))
        grad_fwd = reshape(grad_fwd_flat, size(path))

        @test isapprox(grad_enzyme, grad_fwd; rtol=rtol, atol=atol)
        @test isapprox(grad_enzyme, grad_fd;  rtol=rtol, atol=atol)
    end
end
