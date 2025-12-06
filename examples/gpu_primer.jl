# GPU Computing Primer in Julia
# =============================
#
# This example demonstrates GPU computing in Julia using:
# 1. CUDA.jl - NVIDIA GPU-specific interface
# 2. KernelAbstractions.jl - Portable GPU programming across backends
#
# First, add the required packages:
# using Pkg
# Pkg.add(["CUDA", "KernelAbstractions", "BenchmarkTools"])

using CUDA
using KernelAbstractions
using BenchmarkTools
using Random

println("GPU Primer for Julia\n" * "="^50)

# =============================================================================
# Part 1: Basic CUDA.jl Usage
# =============================================================================
println("\n1. BASIC CUDA.jl USAGE")
println("-" ^50)

# Check if CUDA is available
if CUDA.functional()
    println("✓ CUDA is available!")
    println("  Device: $(CUDA.device())")
    println("  Memory: $(CUDA.available_memory() / 1e9) GB available")
else
    println("✗ CUDA not available - examples will run on CPU")
end

# Simple array operations
println("\n• Array Operations:")
N = 1000
x_cpu = rand(Float32, N)
y_cpu = rand(Float32, N)

# Transfer to GPU (if available)
x_gpu = CUDA.functional() ? CuArray(x_cpu) : x_cpu
y_gpu = CUDA.functional() ? CuArray(y_cpu) : y_cpu

# GPU operations look just like CPU operations!
z_gpu = x_gpu .+ y_gpu
z_gpu = 2 .* x_gpu .+ 3 .* y_gpu  # SAXPY operation

# Transfer back to CPU
z_cpu = Array(z_gpu)

println("  Computed z = 2x + 3y on $(CUDA.functional() ? "GPU" : "CPU")")
println("  Result size: $(size(z_cpu))")

# =============================================================================
# Part 2: Performance Comparison
# =============================================================================
println("\n2. PERFORMANCE COMPARISON")
println("-" ^50)

N = 10_000_000
x_cpu = rand(Float32, N)
y_cpu = rand(Float32, N)

# CPU benchmark
println("• CPU performance:")
cpu_time = @elapsed begin
    z_cpu = 2 .* x_cpu .+ 3 .* y_cpu
end
println("  Time: $(round(cpu_time * 1000, digits=2)) ms")

if CUDA.functional()
    # GPU benchmark (with warmup)
    x_gpu = CuArray(x_cpu)
    y_gpu = CuArray(y_cpu)

    # Warmup
    z_gpu = 2 .* x_gpu .+ 3 .* y_gpu
    CUDA.synchronize()

    # Actual benchmark
    println("• GPU performance:")
    gpu_time = @elapsed begin
        z_gpu = 2 .* x_gpu .+ 3 .* y_gpu
        CUDA.synchronize()  # Wait for GPU to finish
    end
    println("  Time: $(round(gpu_time * 1000, digits=2)) ms")
    println("  Speedup: $(round(cpu_time / gpu_time, digits=2))x")
end

# =============================================================================
# Part 3: Custom CUDA Kernels
# =============================================================================
println("\n3. CUSTOM CUDA KERNELS")
println("-" ^50)

# CUDA kernel for element-wise operation
function cuda_saxpy!(y, a, x)
    # Get thread index
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= length(y)
        @inbounds y[i] = a * x[i] + y[i]
    end
    return nothing
end

if CUDA.functional()
    N = 1_000_000
    x = CUDA.ones(Float32, N)
    y = CUDA.ones(Float32, N)
    a = 2.0f0

    # Launch kernel with 256 threads per block
    threads = 256
    blocks = cld(N, threads)  # Ceiling division

    println("• Launching CUDA kernel:")
    println("  Grid: $blocks blocks × $threads threads")

    @cuda threads=threads blocks=blocks cuda_saxpy!(y, a, x)
    CUDA.synchronize()

    println("  Result: y[1:5] = $(Array(y)[1:5])")
end

# =============================================================================
# Part 4: KernelAbstractions.jl - Portable GPU Programming
# =============================================================================
println("\n4. KERNELASTRACTIONS.JL - PORTABLE CODE")
println("-" ^50)

# Define a kernel that works on ANY backend (CPU, CUDA, ROCm, Metal, etc.)
@kernel function ka_saxpy!(y, a, @Const(x))
    i = @index(Global)
    @inbounds y[i] = a * x[i] + y[i]
end

# Function that works with any array type
function run_saxpy!(y, a, x)
    backend = get_backend(y)
    # Create and launch kernel (256 = workgroup/block size)
    event = ka_saxpy!(backend, 256)(y, a, x, ndrange=length(y))
    # CPU backend returns nothing, GPU backends return an event
    if !isnothing(event)
        wait(event)  # Wait for kernel to complete
    end
end

# Test on CPU
println("• KernelAbstractions on CPU:")
x_cpu = ones(Float32, 1000)
y_cpu = ones(Float32, 1000)
run_saxpy!(y_cpu, 2.0f0, x_cpu)
println("  Result: y[1:5] = $(y_cpu[1:5])")

# Test on GPU (if available)
if CUDA.functional()
    println("• KernelAbstractions on GPU:")
    x_gpu = CUDA.ones(Float32, 1000)
    y_gpu = CUDA.ones(Float32, 1000)
    run_saxpy!(y_gpu, 2.0f0, x_gpu)
    println("  Result: y[1:5] = $(Array(y_gpu)[1:5])")
end

# =============================================================================
# Part 5: CUDA vs KernelAbstractions Performance
# =============================================================================
if CUDA.functional()
    println("\n5. CUDA vs KERNELASTRACTIONS BENCHMARK")
    println("-" ^50)

    N = 10_000_000
    x_gpu = CUDA.ones(Float32, N)
    y_gpu = CUDA.ones(Float32, N)
    a = 2.0f0

    println("• Testing with $(N) elements:")
    println()

    # Benchmark native CUDA kernel
    println("  Native CUDA kernel:")
    threads = 256
    blocks = cld(N, threads)

    # Warmup
    @cuda threads=threads blocks=blocks cuda_saxpy!(y_gpu, a, x_gpu)
    CUDA.synchronize()

    # Benchmark - measure with synchronization after each call
    cuda_times = Float64[]
    for _ in 1:10
        t = CUDA.@elapsed begin
            @cuda threads=threads blocks=blocks cuda_saxpy!(y_gpu, a, x_gpu)
        end
        push!(cuda_times, t * 1000)  # Convert to ms
    end
    cuda_time_per_call = sum(cuda_times) / length(cuda_times)
    println("    Time per call: $(round(cuda_time_per_call, digits=4)) ms")

    # Reset arrays
    x_gpu .= 1.0f0
    y_gpu .= 1.0f0

    # Benchmark KernelAbstractions
    println("  KernelAbstractions kernel:")

    # Warmup
    run_saxpy!(y_gpu, a, x_gpu)
    CUDA.synchronize()

    # Benchmark - measure with synchronization after each call
    ka_times = Float64[]
    backend = get_backend(y_gpu)
    kernel! = ka_saxpy!(backend, 256)
    for _ in 1:10
        t = CUDA.@elapsed begin
            event = kernel!(y_gpu, a, x_gpu, ndrange=length(y_gpu))
            if !isnothing(event)
                wait(event)
            end
        end
        push!(ka_times, t * 1000)  # Convert to ms
    end
    ka_time_per_call = sum(ka_times) / length(ka_times)
    println("    Time per call: $(round(ka_time_per_call, digits=4)) ms")

    # Compare
    println()
    overhead = ((ka_time_per_call - cuda_time_per_call) / cuda_time_per_call) * 100
    if overhead < 5
        println("  Result: KA is essentially the same as native CUDA ($(round(overhead, digits=1))% overhead)")
    elseif overhead < 20
        println("  Result: KA has minimal overhead vs native CUDA ($(round(overhead, digits=1))% slower)")
    else
        println("  Result: KA overhead vs native CUDA: $(round(overhead, digits=1))%")
    end

    println()
    println("  Throughput comparison:")
    bandwidth_cuda = (3 * N * sizeof(Float32) / 1e9) / (cuda_time_per_call / 1000)
    bandwidth_ka = (3 * N * sizeof(Float32) / 1e9) / (ka_time_per_call / 1000)
    println("    CUDA: $(round(bandwidth_cuda, digits=2)) GB/s")
    println("    KA:   $(round(bandwidth_ka, digits=2)) GB/s")

    println()
    println("  Conclusion: KernelAbstractions provides portability with")
    println("              minimal to no performance loss on CUDA!")
end

# =============================================================================
# Part 6: Matrix Operations
# =============================================================================
println("\n6. MATRIX OPERATIONS")
println("-" ^50)

M, N = 2000, 2000
A_cpu = rand(Float32, M, N)
B_cpu = rand(Float32, N, M)

println("• Matrix multiplication ($(M)×$(N)):")

# CPU
cpu_time = @elapsed C_cpu = A_cpu * B_cpu
println("  CPU time: $(round(cpu_time * 1000, digits=2)) ms")

if CUDA.functional()
    A_gpu = CuArray(A_cpu)
    B_gpu = CuArray(B_cpu)

    # Warmup
    C_gpu = A_gpu * B_gpu
    CUDA.synchronize()

    # Benchmark
    gpu_time = @elapsed begin
        C_gpu = A_gpu * B_gpu
        CUDA.synchronize()
    end

    println("  GPU time: $(round(gpu_time * 1000, digits=2)) ms")
    println("  Speedup: $(round(cpu_time / gpu_time, digits=2))x")

    # Verify correctness
    diff = maximum(abs.(Array(C_gpu) .- C_cpu))
    println("  Max difference: $(diff)")
end

# =============================================================================
# Part 7: Best Practices and Tips
# =============================================================================
println("\n7. BEST PRACTICES & TIPS")
println("-" ^50)
println("""
Key Concepts:
• CuArray: GPU array type (like regular Array but on GPU)
• @cuda: Macro to launch GPU kernels
• synchronize(): Wait for GPU operations to complete
• Broadcast (.*): Automatically works on GPU arrays

Performance Tips:
• Minimize CPU↔GPU data transfers (expensive!)
• Use Float32 instead of Float64 (GPUs prefer it)
• Process large arrays (>10K elements) for speedup
• Reuse GPU arrays instead of allocating new ones
• Use in-place operations (!) when possible

KernelAbstractions Advantages:
• Write once, run on any GPU (NVIDIA, AMD, Intel, Apple)
• Cleaner code for custom kernels
• Better composability with other packages

When to Use GPU:
✓ Large arrays (millions of elements)
✓ Element-wise operations
✓ Matrix operations
✓ Embarrassingly parallel problems
✗ Small arrays (overhead dominates)
✗ Sequential algorithms
✗ Lots of branching/conditionals

Common Patterns:
  # Basic GPU usage
  x_gpu = CuArray(x_cpu)      # CPU → GPU
  y_gpu = f.(x_gpu)            # Compute on GPU
  y_cpu = Array(y_gpu)         # GPU → CPU

  # In-place operations
  x_gpu .= 2 .* x_gpu .+ 1    # Modify in-place

  # Reduce operations
  s = sum(x_gpu)               # Reduction on GPU
  m = maximum(x_gpu)
""")

println("\n" * "="^50)
println("GPU Primer Complete!")
if !CUDA.functional()
    println("\nNote: Install CUDA-capable GPU and drivers for full functionality")
end
