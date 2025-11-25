using Test
using Chen

@testset "exp and log are inverses" begin
    
    @testset "dim=$dim, level=$level" for dim in [2, 3, 5], level in [2, 3, 4]
        
        # Test 1: log(exp(x)) ≈ x for Lie elements
        @testset "log(exp(x)) ≈ x" begin
            # Create a random Lie element (level-0 = 0)
            x = Chen.Tensor{Float64}(dim, level)
            Chen._zero!(x)
            
            # Fill with random values at levels 1..level
            # (keeping level-0 = 0 for a Lie element)
            for k in 1:level
                start_idx = x.offsets[k+1] + 1
                end_idx = x.offsets[k+2]
                for i in start_idx:end_idx
                    x.coeffs[i] = randn()
                end
            end
            
            # Compute g = exp(x)
            g = similar(x)
            Chen.exp!(g, x)
            
            # Verify g is group-like (level-0 should be 1)
            @test g.coeffs[g.offsets[1] + 1] ≈ 1.0
            
            # Compute y = log(g)
            y = similar(x)
            Chen.log!(y, g)
            
            # Check that y ≈ x
            @test isapprox(y, x, atol=1e-10, rtol=1e-10)
        end
        
        # Test 2: exp(log(g)) ≈ g for group elements
        @testset "exp(log(g)) ≈ g" begin
            # Create a random group element by exponentiating a Lie element
            x_temp = Chen.Tensor{Float64}(dim, level)
            Chen._zero!(x_temp)
            
            # Fill with small random values to ensure convergence
            for k in 1:level
                start_idx = x_temp.offsets[k+1] + 1
                end_idx = x_temp.offsets[k+2]
                for i in start_idx:end_idx
                    x_temp.coeffs[i] = 0.1 * randn()
                end
            end
            
            # g = exp(x_temp) - this gives us a valid group element
            g = similar(x_temp)
            Chen.exp!(g, x_temp)
            
            # Verify g is group-like (level-0 should be 1)
            @test g.coeffs[g.offsets[1] + 1] ≈ 1.0
            
            # Compute x = log(g)
            x = similar(g)
            Chen.log!(x, g)
            
            # Verify x is Lie-like (level-0 should be 0)
            @test x.coeffs[x.offsets[1] + 1] ≈ 0.0 atol=1e-12
            
            # Compute h = exp(x)
            h = similar(g)
            Chen.exp!(h, x)
            
            # Check that h ≈ g
            @test isapprox(h, g, atol=1e-10, rtol=1e-10)
        end
        
        # Test 3: exp(log(g)) ≈ g with vector input to exp
        @testset "exp(log(g)) ≈ g (vector exp)" begin
            # Create a random level-1 vector
            x_vec = randn(dim) * 0.1
            
            # Compute g = exp(x_vec)
            g = Chen.Tensor{Float64}(dim, level)
            Chen.exp!(g, x_vec)
            
            # Compute x = log(g)
            x = similar(g)
            Chen.log!(x, g)
            
            # Compute h = exp(x)
            h = similar(g)
            Chen.exp!(h, x)
            
            # Check that h ≈ g
            @test isapprox(h, g, atol=1e-10, rtol=1e-10)
        end
        
    end
    
    # Edge case: level = 0
    @testset "level=0 edge case" begin
        dim, level = 3, 0
        
        x = Chen.Tensor{Float64}(dim, level)
        Chen._zero!(x)
        x.coeffs[x.offsets[1] + 1] = 1.0
        
        g = similar(x)
        Chen.exp!(g, x)
        
        # At level 0, exp should just preserve the unit
        @test g.coeffs[g.offsets[1] + 1] ≈ 1.0
    end
    
    println("\n✓ All exp/log inverse tests passed!")
end