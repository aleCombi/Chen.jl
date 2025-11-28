using ChenSignatures
using BenchmarkTools
using StaticArrays
using Printf

function benchmark_three_versions()
    println("="^80)
    println("Benchmark: sig vs sig_enzyme vs signature_path (SVector)")
    println("="^80)
    
    test_cases = [
        (N=100, D=2, m=3, name="N=100, D=2, m=3"),
        (N=500, D=2, m=3, name="N=500, D=2, m=3"),
        (N=1000, D=2, m=3, name="N=1000, D=2, m=3"),
        (N=100, D=3, m=4, name="N=100, D=3, m=4"),
        (N=500, D=3, m=4, name="N=500, D=3, m=4"),
        (N=1000, D=3, m=3, name="N=1000, D=3, m=3"),
    ]
    
    for case in test_cases
        println("\n" * case.name)
        println("-"^80)
        
        # Generate random path as matrix
        path_matrix = randn(case.N, case.D)
        
        # Convert to vector of SVectors
        path_svec = [SVector{case.D}(path_matrix[i,:]) for i in 1:case.N]
        
        # Benchmark 1: sig (goes through SVector conversion internally)
        print("sig(matrix):          ")
        t_sig = @belapsed sig($path_matrix, $(case.m))
        
        # Benchmark 2: signature_path with pre-converted SVectors
        print("signature_path(svec): ")
        t_sigpath = @belapsed begin
            tensor = signature_path(Tensor{Float64}, $path_svec, $(case.m))
            ChenSignatures._flatten_tensor(tensor)
        end
        
        # Benchmark 3: sig_enzyme (Vector-based, no SVector)
        print("sig_enzyme(matrix):   ")
        t_enzyme = try
            @belapsed sig_enzyme($path_matrix, $(case.m))
        catch e
            println("Not supported (D=$(case.D), m=$(case.m))")
            continue
        end
        
        # Results
        @printf("  sig(matrix):          %.3f ms\n", t_sig * 1e3)
        @printf("  signature_path(svec): %.3f ms\n", t_sigpath * 1e3)
        @printf("  sig_enzyme(matrix):   %.3f ms\n", t_enzyme * 1e3)
        
        @printf("  Speedup sig_enzyme vs sig:           %.2fx\n", t_sig / t_enzyme)
        @printf("  Speedup sig_enzyme vs signature_path: %.2fx\n", t_sigpath / t_enzyme)
        
        # Verify all give same result
        result_sig = sig(path_matrix, case.m)
        result_sigpath = begin
            tensor = signature_path(Tensor{Float64}, path_svec, case.m)
            ChenSignatures._flatten_tensor(tensor)
        end
        result_enzyme = sig_enzyme(path_matrix, case.m)
        
        all_match = isapprox(result_sig, result_sigpath, rtol=1e-10) && 
                    isapprox(result_sig, result_enzyme, rtol=1e-10)
        
        if all_match
            println("  ✓ All results match")
        else
            println("  ⚠ Results differ!")
        end
    end
    
    println("\n" * "="^80)
    println("\nConclusion:")
    println("  - sig(matrix): Converts to SVector internally, then calls signature_path")
    println("  - signature_path(svec): Direct computation with pre-converted SVectors")
    println("  - sig_enzyme(matrix): Direct Vector-based computation, no conversions")
end

benchmark_three_versions()