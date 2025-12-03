using ChenSignatures
using StaticArrays

function benchmark_add_scaled(D, M, iterations=10000)
    dest = ChenSignatures.Tensor{Float64,D,M}()
    src = ChenSignatures.Tensor{Float64,D,M}()

    # Fill with random data
    for i in 1:length(dest.coeffs)
        dest.coeffs[i] = rand()
        src.coeffs[i] = rand()
    end
    α = 0.5

    # Warmup
    for _ in 1:100
        ChenSignatures.add_scaled!(dest, src, α)
    end

    # Benchmark
    start = time_ns()
    for _ in 1:iterations
        ChenSignatures.add_scaled!(dest, src, α)
    end
    elapsed = (time_ns() - start) / 1e9

    return elapsed / iterations * 1e6  # microseconds per iteration
end

println("=" ^ 60)
println("add_scaled! Performance Benchmark (@simd version)")
println("=" ^ 60)
println()

configs = [
    (2, 2, 100000),
    (2, 4, 50000),
    (3, 3, 50000),
    (3, 4, 20000),
    (4, 4, 10000),
    (5, 5, 5000),
]

for (D, M, iters) in configs
    time_us = benchmark_add_scaled(D, M, iters)
    tensor_size = D^M
    println("D=$D, M=$M (size=$tensor_size): $(round(time_us, digits=3)) μs/call")
end

println()
println("Testing log! which uses add_scaled!...")

function benchmark_log(D, M, iterations=1000)
    g = ChenSignatures.Tensor{Float64,D,M}()
    out = similar(g)

    # Create a valid group element
    ChenSignatures._zero!(g)
    g.coeffs[g.offsets[1] + 1] = 1.0
    for k in 1:M
        len = D^k
        start = g.offsets[k+1] + 1
        for i in 0:(len-1)
            g.coeffs[start + i] = 0.01 * randn()
        end
    end

    # Warmup
    for _ in 1:10
        ChenSignatures.log!(out, g)
    end

    # Benchmark
    start = time_ns()
    for _ in 1:iterations
        ChenSignatures.log!(out, g)
    end
    elapsed = (time_ns() - start) / 1e9

    return elapsed / iterations * 1e6  # microseconds
end

log_configs = [
    (2, 3, 5000),
    (2, 4, 2000),
    (3, 3, 2000),
    (3, 4, 1000),
]

for (D, M, iters) in log_configs
    time_us = benchmark_log(D, M, iters)
    println("log! D=$D, M=$M: $(round(time_us, digits=2)) μs/call")
end

println()
println("=" ^ 60)
println("Benchmark complete")
println("=" ^ 60)
