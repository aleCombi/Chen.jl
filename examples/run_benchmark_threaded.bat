@echo off
cd /d c:\repos\ChenSignatures.jl\examples
set JULIA_NUM_THREADS=4
julia --project=. benchmark_comprehensive.jl
