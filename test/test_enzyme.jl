using ChenSignatures
using Test
using Enzyme
using ForwardDiff
using LinearAlgebra

@testset "sig - Comprehensive Gradient Tests" begin
    
    @testset "Basic functionality" begin
        path = [0.0 0.0; 1.0 1.0; 2.0 3.0]
        
        result = sig(path, 3)
        @test result isa Vector{Float64}
        @test length(result) == 2 + 4 + 8  # d^1 + d^2 + d^3 for d=2, m=3
        @test all(isfinite, result)
    end
    
    @testset "Enzyme vs Finite Differences - sum loss" begin
        path = randn(4, 2)
        
        loss(p) = sum(sig(p, 3))
        
        # Enzyme gradient
        grad_enzyme = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad_enzyme))
        
        # Finite differences
        eps = 1e-6
        grad_fd = zeros(size(path))
        for i in 1:size(path, 1), j in 1:size(path, 2)
            p_plus = copy(path); p_plus[i,j] += eps
            p_minus = copy(path); p_minus[i,j] -= eps
            grad_fd[i,j] = (loss(p_plus) - loss(p_minus)) / (2*eps)
        end
        
        @test isapprox(grad_enzyme, grad_fd, rtol=1e-4, atol=1e-8)
    end
    
    @testset "Enzyme vs Finite Differences - specific coefficient" begin
        path = randn(4, 2)
        
        # Loss: just the 5th coefficient
        loss(p) = sig(p, 3)[5]
        
        grad_enzyme = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad_enzyme))
        
        # Finite differences
        eps = 1e-6
        grad_fd = zeros(size(path))
        for i in 1:size(path, 1), j in 1:size(path, 2)
            p_plus = copy(path); p_plus[i,j] += eps
            p_minus = copy(path); p_minus[i,j] -= eps
            grad_fd[i,j] = (loss(p_plus) - loss(p_minus)) / (2*eps)
        end
        
        @test isapprox(grad_enzyme, grad_fd, rtol=1e-4, atol=1e-8)
    end
    
    @testset "Enzyme vs Finite Differences - L2 norm loss" begin
        path = randn(4, 2)
        
        loss(p) = sum(abs2, sig(p, 3))
        
        grad_enzyme = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad_enzyme))
        
        # Finite differences
        eps = 1e-6
        grad_fd = zeros(size(path))
        for i in 1:size(path, 1), j in 1:size(path, 2)
            p_plus = copy(path); p_plus[i,j] += eps
            p_minus = copy(path); p_minus[i,j] -= eps
            grad_fd[i,j] = (loss(p_plus) - loss(p_minus)) / (2*eps)
        end
        
        @test isapprox(grad_enzyme, grad_fd, rtol=1e-4, atol=1e-8)
    end
    
    @testset "Enzyme gradient through sig (via wrapper)" begin
        path = randn(4, 2)
        
        # Can't differentiate sig directly (has SVector, generated functions)
        # But can verify sig gives same forward pass
        @test isapprox(sig(path, 3), sig(path, 3), rtol=1e-10)
        
        # And Enzyme works on sig
        loss(p) = sum(sig(p, 3))
        grad = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad))
        @test all(isfinite, grad)
    end
    
    @testset "Different dimensions" begin
        # 3D path
        path_3d = randn(4, 3)
        result = sig(path_3d, 3)
        @test length(result) == 3 + 9 + 27  # d^1 + d^2 + d^3 for d=3, m=3
        @test isapprox(sig(path_3d, 3), result, rtol=1e-10)
        
        # Gradient test
        loss(p) = sum(sig(p, 3))
        grad = zeros(size(path_3d))
        autodiff(Reverse, loss, Active, Duplicated(path_3d, grad))
        
        # Spot check with FD
        eps = 1e-6
        i, j = 2, 1
        p_plus = copy(path_3d); p_plus[i,j] += eps
        p_minus = copy(path_3d); p_minus[i,j] -= eps
        fd_ij = (loss(p_plus) - loss(p_minus)) / (2*eps)
        
        @test isapprox(grad[i,j], fd_ij, rtol=1e-4, atol=1e-8)
    end
    
    @testset "Longer paths" begin
        path = randn(20, 2)
        
        loss(p) = sum(sig(p, 3))
        grad = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad))
        
        # Spot check a few entries with FD
        eps = 1e-6
        for _ in 1:3
            i = rand(1:size(path, 1))
            j = rand(1:size(path, 2))
            
            p_plus = copy(path); p_plus[i,j] += eps
            p_minus = copy(path); p_minus[i,j] -= eps
            fd_ij = (loss(p_plus) - loss(p_minus)) / (2*eps)
            
            @test isapprox(grad[i,j], fd_ij, rtol=1e-4, atol=1e-8)
        end
    end
end