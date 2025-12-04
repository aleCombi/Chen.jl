using ChenSignatures
using ChainRulesCore
using Zygote
using Enzyme
using ForwardDiff
using FiniteDifferences
using Test
using LinearAlgebra
using Random

@testset "ChainRules rrule for sig" begin
    Random.seed!(42)
    
    @testset "Basic rrule functionality" begin
        # Skip on Julia 1.12+ (rrule uses Enzyme internally, experimental support)
        if VERSION >= v"1.12"
            @test_skip "Skipping rrule test on Julia $(VERSION)"
            return
        end

        path = [0.0 0.0; 1.0 1.0; 2.0 3.0]
        m = 3

        # Test forward pass
        result, pullback = rrule(sig, path, m)
        @test result isa Vector{Float64}
        @test length(result) == 2 + 4 + 8  # d^1 + d^2 + d^3
        @test all(isfinite, result)

        # Test pullback
        ȳ = ones(length(result))
        ∂self, ∂path, ∂m = pullback(ȳ)
        @test ∂self isa NoTangent
        @test ∂m isa NoTangent
        @test ∂path isa Matrix{Float64}
        @test size(∂path) == size(path)
        @test all(isfinite, ∂path)
    end
    
    @testset "Zygote integration - sum loss" begin
        # Skip Enzyme comparison on Julia 1.12+ (experimental support)
        if VERSION >= v"1.12"
            @test_skip "Skipping Enzyme comparison on Julia $(VERSION)"
            return
        end

        path = randn(4, 2)

        # Compute gradient with Zygote
        grad_zygote = Zygote.gradient(p -> sum(sig(p, 3)), path)[1]

        # Compute gradient with Enzyme (baseline)
        grad_enzyme = zeros(size(path))
        autodiff(Reverse, p -> sum(sig(p, 3)), Active, Duplicated(path, grad_enzyme))

        @test isapprox(grad_zygote, grad_enzyme; rtol=1e-10, atol=1e-12)
    end
    
    @testset "Zygote integration - specific coefficient" begin
        # Skip Enzyme comparison on Julia 1.12+ (experimental support)
        if VERSION >= v"1.12"
            @test_skip "Skipping Enzyme comparison on Julia $(VERSION)"
            return
        end

        path = randn(4, 2)
        idx = 5

        grad_zygote = Zygote.gradient(p -> sig(p, 3)[idx], path)[1]

        grad_enzyme = zeros(size(path))
        autodiff(Reverse, p -> sig(p, 3)[idx], Active, Duplicated(path, grad_enzyme))

        @test isapprox(grad_zygote, grad_enzyme; rtol=1e-10, atol=1e-12)
    end

    @testset "Zygote integration - L2 loss" begin
        # Skip Enzyme comparison on Julia 1.12+ (experimental support)
        if VERSION >= v"1.12"
            @test_skip "Skipping Enzyme comparison on Julia $(VERSION)"
            return
        end

        path = randn(4, 2)

        grad_zygote = Zygote.gradient(p -> sum(abs2, sig(p, 3)), path)[1]

        grad_enzyme = zeros(size(path))
        autodiff(Reverse, p -> sum(abs2, sig(p, 3)), Active, Duplicated(path, grad_enzyme))

        @test isapprox(grad_zygote, grad_enzyme; rtol=1e-10, atol=1e-12)
    end
    
    @testset "Comparison with ForwardDiff" begin
        path = randn(3, 2)
        
        # Zygote via rrule
        grad_zygote = Zygote.gradient(p -> sum(sig(p, 3)), path)[1]
        
        # ForwardDiff (requires flattening)
        loss_flat(x) = sum(sig(reshape(x, size(path)), 3))
        grad_fwd_flat = ForwardDiff.gradient(loss_flat, vec(path))
        grad_fwd = reshape(grad_fwd_flat, size(path))
        
        @test isapprox(grad_zygote, grad_fwd; rtol=1e-10, atol=1e-12)
    end
    
    @testset "Comparison with Finite Differences" begin
        path = randn(3, 2)
        eps = 1e-6
        
        grad_zygote = Zygote.gradient(p -> sum(sig(p, 3)), path)[1]
        
        # Central finite differences
        grad_fd = zeros(size(path))
        for i in 1:size(path, 1), j in 1:size(path, 2)
            p_plus = copy(path)
            p_plus[i, j] += eps
            p_minus = copy(path)
            p_minus[i, j] -= eps
            grad_fd[i, j] = (sum(sig(p_plus, 3)) - sum(sig(p_minus, 3))) / (2 * eps)
        end
        
        @test isapprox(grad_zygote, grad_fd; rtol=1e-4, atol=1e-8)
    end
    
    @testset "Different path dimensions" begin
        # 2D paths
        path_2d = randn(5, 2)
        grad_2d = Zygote.gradient(p -> sum(sig(p, 3)), path_2d)[1]
        @test size(grad_2d) == size(path_2d)
        @test all(isfinite, grad_2d)
        
        # 3D paths
        path_3d = randn(4, 3)
        grad_3d = Zygote.gradient(p -> sum(sig(p, 3)), path_3d)[1]
        @test size(grad_3d) == size(path_3d)
        @test all(isfinite, grad_3d)
        
        # 4D paths
        path_4d = randn(4, 4)
        grad_4d = Zygote.gradient(p -> sum(sig(p, 2)), path_4d)[1]
        @test size(grad_4d) == size(path_4d)
        @test all(isfinite, grad_4d)
    end
    
    @testset "Different truncation levels" begin
        path = randn(4, 2)
        
        for m in 1:4
            grad = Zygote.gradient(p -> sum(sig(p, m)), path)[1]
            @test size(grad) == size(path)
            @test all(isfinite, grad)
        end
    end
    
    @testset "Longer paths" begin
        # Skip Enzyme comparison on Julia 1.12+ (experimental support)
        if VERSION >= v"1.12"
            @test_skip "Skipping Enzyme comparison on Julia $(VERSION)"
            return
        end

        path = randn(20, 2)

        grad_zygote = Zygote.gradient(p -> sum(sig(p, 3)), path)[1]

        grad_enzyme = zeros(size(path))
        autodiff(Reverse, p -> sum(sig(p, 3)), Active, Duplicated(path, grad_enzyme))

        @test isapprox(grad_zygote, grad_enzyme; rtol=1e-10, atol=1e-12)
    end
    
    @testset "Edge cases" begin
        # Minimal path (N = 2)
        path_min = randn(2, 2)
        grad_min = Zygote.gradient(p -> sum(sig(p, 3)), path_min)[1]
        @test all(isfinite, grad_min)
        
        # Zero path
        path_zero = zeros(4, 2)
        grad_zero = Zygote.gradient(p -> sum(sig(p, 3)), path_zero)[1]
        @test all(isfinite, grad_zero)
    end
    
    @testset "Gradient spot checks" begin
        path = randn(10, 2)
        eps = 1e-6
        
        grad_zygote = Zygote.gradient(p -> sum(sig(p, 3)), path)[1]
        
        # Check a few random entries
        for _ in 1:5
            i = rand(1:size(path, 1))
            j = rand(1:size(path, 2))
            
            p_plus = copy(path)
            p_plus[i, j] += eps
            p_minus = copy(path)
            p_minus[i, j] -= eps
            
            fd_ij = (sum(sig(p_plus, 3)) - sum(sig(p_minus, 3))) / (2 * eps)
            
            @test isapprox(grad_zygote[i, j], fd_ij; rtol=1e-4, atol=1e-8)
        end
    end
end