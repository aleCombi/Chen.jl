using Test
using Chen
using StaticArrays

# Helper to run tests for specific D and M types
function test_exp_log_dim(::Val{D}, ::Val{M}) where {D,M}
    
    @testset "D=$D, M=$M" begin
        # Test 1: log(exp(x)) ≈ x for Lie elements
        @testset "log(exp(x)) ≈ x" begin
            # Create a Lie element-like tensor (level-0 coefficient is 0)
            x = Chen.Tensor{Float64,D,M}()
            Chen._zero!(x) 
            
            # Fill with random values at levels 1..M
            for k in 1:M
                start_idx = x.offsets[k+1] + 1
                end_idx = x.offsets[k+2]
                for i in start_idx:end_idx
                    x.coeffs[i] = randn()
                end
            end
            
            # Compute g = exp(x)
            g = similar(x)
            Chen.exp!(g, x) # This requires the new implementation in src/dense_tensors.jl
            
            # Verify g is group-like (level-0 should be 1)
            @test g.coeffs[g.offsets[1] + 1] ≈ 1.0
            
            # Compute y = log(g)
            y = similar(x)
            Chen.log!(y, g)
            
            # Check that y ≈ x
            @test x.coeffs ≈ y.coeffs atol=1e-10 rtol=1e-10
        end
        
        # Test 2: exp(log(g)) ≈ g for group elements
        @testset "exp(log(g)) ≈ g" begin
            # Create a random group element
            x_temp = Chen.Tensor{Float64,D,M}()
            Chen._zero!(x_temp)
            for k in 1:M
                start_idx = x_temp.offsets[k+1] + 1
                end_idx = x_temp.offsets[k+2]
                for i in start_idx:end_idx; x_temp.coeffs[i] = 0.1 * randn(); end
            end
            
            g = similar(x_temp)
            Chen.exp!(g, x_temp) 
            
            # Compute x = log(g)
            x = similar(g)
            Chen.log!(x, g)
            
            # Compute h = exp(x)
            h = similar(g)
            Chen.exp!(h, x)
            
            # Check that h ≈ g
            @test h.coeffs ≈ g.coeffs atol=1e-10 rtol=1e-10
        end
        
        # Test 3: exp(log(g)) ≈ g with vector input to exp
        @testset "exp(vector) consistency" begin
            x_vec = SVector{D, Float64}(randn(D) * 0.1)
            
            # Compute g = exp(x_vec) directly (uses SVector optimized exp!)
            g = Chen.Tensor{Float64,D,M}()
            Chen.exp!(g, x_vec)
            
            # Compute x = log(g)
            x = similar(g)
            Chen.log!(x, g)
            
            # Compute h = exp(x) (uses Tensor generic exp!)
            h = similar(g)
            Chen.exp!(h, x)
            
            @test h.coeffs ≈ g.coeffs atol=1e-10 rtol=1e-10
        end
    end
end

@testset "exp and log are inverses" begin
    for d in [2, 3], m in [2, 3]
        test_exp_log_dim(Val(d), Val(m))
    end
end