using Test
using ChenSignatures
using StaticArrays

@testset "add_scaled! skips padding correctly" begin
    @testset "D=$D, M=$M" for D in [2, 3], M in [2, 3, 4]
        # Create test tensors
        dest = ChenSignatures.Tensor{Float64,D,M}()
        src = ChenSignatures.Tensor{Float64,D,M}()

        # Fill with known values (only meaningful data, not padding)
        ChenSignatures._zero!(dest)
        ChenSignatures._zero!(src)

        # Set level 0
        dest.coeffs[dest.offsets[1] + 1] = 1.0
        src.coeffs[src.offsets[1] + 1] = 2.0

        # Set level 1 and above with different values
        for k in 1:M
            len = D^k
            start = dest.offsets[k+1] + 1
            for i in 0:(len-1)
                dest.coeffs[start + i] = Float64(k)
                src.coeffs[start + i] = Float64(k + 10)
            end
        end

        # Store original padding values (if any exist between levels)
        # Padding should be in positions that aren't touched by the level iterations
        padding_positions = Int[]
        for k in 1:M
            prev_end = k == 1 ? dest.offsets[1] + 1 : dest.offsets[k] + D^(k-1)
            curr_start = dest.offsets[k+1]
            for pos in (prev_end+1):curr_start
                push!(padding_positions, pos)
            end
        end

        # Set padding to a sentinel value to verify it's not touched
        sentinel = -999.0
        for pos in padding_positions
            dest.coeffs[pos] = sentinel
            src.coeffs[pos] = sentinel
        end

        # Perform add_scaled! with α = 0.5
        α = 0.5
        ChenSignatures.add_scaled!(dest, src, α)

        # Verify level 0
        @test dest.coeffs[dest.offsets[1] + 1] ≈ 1.0 + α * 2.0

        # Verify all meaningful levels
        for k in 1:M
            len = D^k
            start = dest.offsets[k+1] + 1
            for i in 0:(len-1)
                expected = Float64(k) + α * Float64(k + 10)
                @test dest.coeffs[start + i] ≈ expected
            end
        end

        # CRITICAL: Verify padding was NOT touched
        for pos in padding_positions
            if dest.coeffs[pos] != sentinel
                error("Padding at position $pos was modified! Expected $sentinel, got $(dest.coeffs[pos]). This means add_scaled! is touching padding.")
            end
            @test dest.coeffs[pos] == sentinel
        end
    end

    @testset "Correctness: compare with manual computation" begin
        D, M = 2, 3
        dest = ChenSignatures.Tensor{Float64,D,M}()
        src = ChenSignatures.Tensor{Float64,D,M}()

        # Random values
        for k in 0:M
            if k == 0
                dest.coeffs[dest.offsets[1] + 1] = randn()
                src.coeffs[src.offsets[1] + 1] = randn()
            else
                len = D^k
                start = dest.offsets[k+1] + 1
                for i in 0:(len-1)
                    dest.coeffs[start + i] = randn()
                    src.coeffs[start + i] = randn()
                end
            end
        end

        # Copy for manual computation
        dest_copy = copy(dest.coeffs)
        α = 2.5

        # Apply add_scaled!
        ChenSignatures.add_scaled!(dest, src, α)

        # Manually compute expected result for each level
        @test dest.coeffs[dest.offsets[1] + 1] ≈ dest_copy[dest.offsets[1] + 1] + α * src.coeffs[src.offsets[1] + 1]

        for k in 1:M
            len = D^k
            start = dest.offsets[k+1] + 1
            for i in 0:(len-1)
                expected = dest_copy[start + i] + α * src.coeffs[start + i]
                @test dest.coeffs[start + i] ≈ expected
            end
        end
    end
end
