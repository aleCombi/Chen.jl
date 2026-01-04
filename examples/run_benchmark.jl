# Benchmark Launcher with Automatic Threading
# Ensures Julia runs with 4 threads for proper multi-threaded benchmarking

if Threads.nthreads() < 4
    println("⚠ Running with only $(Threads.nthreads()) thread(s)")
    println("Re-launching with 4 threads for optimal performance...")
    println()

    # Re-launch Julia with 4 threads
    script_path = joinpath(@__DIR__, basename(PROGRAM_FILE))
    julia_cmd = `$(Base.julia_cmd()) -t 4 --project=$(@__DIR__) $(script_path)`
    run(julia_cmd)
    exit(0)
else
    println("✓ Running with $(Threads.nthreads()) threads")
    println()

    # Run the actual benchmark
    include("benchmark_comprehensive.jl")
end
