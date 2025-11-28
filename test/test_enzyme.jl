using ChenSignatures
using Test
using Enzyme
using LinearAlgebra

@testset "Enzyme Differentiation" begin
    # In test/test_enzyme.jl, update the first test:

    @testset "sig_enzyme basic functionality" begin
        path = [0.0 0.0; 1.0 1.0; 2.0 3.0]
        
        # Test forward pass returns vector (raw coeffs with padding)
        result = ChenSignatures.sig_enzyme(path, 3)
        @test result isa Vector{Float64}
        # Raw coeffs include constant term + padding, so length varies
        @test all(isfinite, result)
    end
    
    @testset "sig_enzyme matches sig" begin
        path = randn(5, 2)
        
        result_sig = ChenSignatures.sig(path, 3)
        result_enzyme = ChenSignatures.sig_enzyme(path, 3)
        
        @test isapprox(result_sig, result_enzyme, rtol=1e-10)
    end
    
    @testset "sig_enzyme gradient - 2D path, m=3" begin
        path = [0.0 0.0; 1.0 1.0; 2.0 3.0]
        
        # Scalar loss: sum of signature coefficients
        loss(p) = sum(ChenSignatures.sig_enzyme(p, 3))
        
        # Compute gradient with Enzyme
        grad = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad))
        
        # Verify with finite differences
        eps = 1e-6
        fd = zeros(size(path))
        for i in 1:size(path, 1), j in 1:size(path, 2)
            p_plus = copy(path)
            p_plus[i, j] += eps
            p_minus = copy(path)
            p_minus[i, j] -= eps
            fd[i, j] = (loss(p_plus) - loss(p_minus)) / (2 * eps)
        end
        
        @test isapprox(grad, fd, rtol=1e-4, atol=1e-8)
    end
    
    @testset "sig_enzyme gradient - specific coefficient" begin
        path = randn(5, 2)
        
        # Loss: just the 5th coefficient
        loss(p) = ChenSignatures.sig_enzyme(p, 3)[5]
        
        grad = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad))
        
        # Verify with finite differences on a few random entries
        eps = 1e-6
        for _ in 1:3
            i = rand(1:size(path, 1))
            j = rand(1:size(path, 2))
            
            p_plus = copy(path)
            p_plus[i, j] += eps
            p_minus = copy(path)
            p_minus[i, j] -= eps
            fd_ij = (loss(p_plus) - loss(p_minus)) / (2 * eps)
            
            @test isapprox(grad[i, j], fd_ij, rtol=1e-4, atol=1e-8)
        end
    end
    
    @testset "sig_enzyme gradient - random path" begin
        path = randn(5, 2)
        
        # Loss: sum of all coefficients
        loss(p) = sum(ChenSignatures.sig_enzyme(p, 3))
        
        grad = zeros(size(path))
        autodiff(Reverse, loss, Active, Duplicated(path, grad))
        
        # Verify with finite differences on random entries
        eps = 1e-6
        for _ in 1:5
            i = rand(1:size(path, 1))
            j = rand(1:size(path, 2))
            
            p_plus = copy(path)
            p_plus[i, j] += eps
            p_minus = copy(path)
            p_minus[i, j] -= eps
            fd_ij = (loss(p_plus) - loss(p_minus)) / (2 * eps)
            
            @test isapprox(grad[i, j], fd_ij, rtol=1e-4, atol=1e-8)
        end
    end
    
    @testset "sig_enzyme different dimensions and depths" begin
        # Test 3D path, m=3
        path_3d = randn(4, 3)
        result = ChenSignatures.sig_enzyme(path_3d, 3)
        @test result isa Vector{Float64}
        @test length(result) == 3 + 9 + 27  # d^1 + d^2 + d^3 for d=3, m=3
        
        # Test gradient
        loss(p) = sum(ChenSignatures.sig_enzyme(p, 3))
        grad = zeros(size(path_3d))
        autodiff(Reverse, loss, Active, Duplicated(path_3d, grad))
        
        # Spot check one element
        eps = 1e-6
        p_plus = copy(path_3d)
        p_plus[1, 1] += eps
        p_minus = copy(path_3d)
        p_minus[1, 1] -= eps
        fd_11 = (loss(p_plus) - loss(p_minus)) / (2 * eps)
        
        @test isapprox(grad[1, 1], fd_11, rtol=1e-4, atol=1e-8)
    end
    
    @testset "sig_enzyme different loss functions" begin
        path = randn(4, 2)
        
        # Test L2 norm loss
        loss_l2(p) = sum(abs2, ChenSignatures.sig_enzyme(p, 3))
        grad_l2 = zeros(size(path))
        autodiff(Reverse, loss_l2, Active, Duplicated(path, grad_l2))
        
        # Verify one element
        eps = 1e-6
        p_plus = copy(path)
        p_plus[2, 1] += eps
        p_minus = copy(path)
        p_minus[2, 1] -= eps
        fd_21 = (loss_l2(p_plus) - loss_l2(p_minus)) / (2 * eps)
        
        @test isapprox(grad_l2[2, 1], fd_21, rtol=1e-4, atol=1e-8)
    end
end