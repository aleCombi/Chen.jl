# GPU Batch Processing using KernelAbstractions.jl
# Provides portable GPU acceleration across CUDA, ROCm, Metal, and CPU backends

using KernelAbstractions

# ============================================================================
# GPU Kernels
# ============================================================================

@kernel function sig_batch_kernel!(
    results,           # (sig_len, B)
    paths,             # (N, D, B)
    @Const(offsets),   # Vector{Int}, length M+2
    ws_B1_all,         # (ws_len, B)
    ws_B2_all,         # (ws_len, B)
    tensor_coeffs_all, # (tensor_len, B)
    @Const(inv_level), # Vector{T}, length M with inv_level[k] = 1/k
    ::Val{D},
    ::Val{M}
) where {D, M}
    batch_idx = @index(Global, Linear)
    B = size(paths, 3)

    if batch_idx <= B
        T = eltype(paths)
        N = size(paths, 1)
        tensor_len = offsets[end]

        # Initialize tensor to identity (level 0 = 1, rest = 0)
        @inbounds for i in 1:tensor_len
            tensor_coeffs_all[i, batch_idx] = zero(T)
        end
        @inbounds tensor_coeffs_all[offsets[1] + 1, batch_idx] = one(T)

        # Process path increments
        @inbounds for i in 1:(N-1)
        # Precompute increment z = path[i+1] - path[i] ONCE
        # Store in tuple (compile-time known size D)
        z = ntuple(d -> paths[i+1, d, batch_idx] - paths[i, d, batch_idx], Val(D))

        # Horner update for levels M down to 2
        @inbounds for k in M:-1:2
            inv_k = inv_level[k]  # Precomputed 1/k

            # Initialize B1 with z * inv_k
            for d in 1:D
                ws_B1_all[d, batch_idx] = z[d] * inv_k
            end

            current_len = D

            # Inner iterations
            for iter in 1:(k-2)
                inv_next = inv_level[k - iter]  # Precomputed 1/(k-iter)
                a_start = offsets[iter + 1]

                # Choose source/destination buffers
                src_is_B1 = isodd(iter)

                for r in 1:current_len
                    src_val = src_is_B1 ? ws_B1_all[r, batch_idx] : ws_B2_all[r, batch_idx]
                    coeff_val = tensor_coeffs_all[a_start + r, batch_idx]
                    val = src_val + coeff_val
                    scaled = val * inv_next

                    base_idx = (r - 1) * D
                    @inbounds for d in 1:D
                        idx = base_idx + d
                        if src_is_B1
                            ws_B2_all[idx, batch_idx] = scaled * z[d]
                        else
                            ws_B1_all[idx, batch_idx] = scaled * z[d]
                        end
                    end
                end

                current_len *= D
            end

            # Final iteration
            last_iter_count = k - 2
            use_B2 = (last_iter_count > 0 && isodd(last_iter_count))
            a_prev_start = offsets[k]
            a_tgt_start = offsets[k + 1]

            for r in 1:current_len
                src_val = use_B2 ? ws_B2_all[r, batch_idx] : ws_B1_all[r, batch_idx]
                coeff_val = tensor_coeffs_all[a_prev_start + r, batch_idx]
                val = src_val + coeff_val

                base_idx = (r - 1) * D
                @inbounds for d in 1:D
                    idx = a_tgt_start + base_idx + d
                    tensor_coeffs_all[idx, batch_idx] += val * z[d]
                end
            end
        end

            # Level 1 update
            start_1 = offsets[2]
            @inbounds for d in 1:D
                tensor_coeffs_all[start_1 + d, batch_idx] += z[d]
            end
        end

        # Flatten tensor to result (remove padding)
        idx_out = 1
        @inbounds for k in 1:M
            start_offset = offsets[k + 1]
            len = D^k
            for j in 1:len
                results[idx_out, batch_idx] = tensor_coeffs_all[start_offset + j, batch_idx]
                idx_out += 1
            end
        end
    end
end

# ============================================================================
# GPU Batch Processing - High-Level API
# ============================================================================

"""
    sig_batch_gpu(paths::AbstractArray{T,3}, m::Int; backend=get_backend(paths)) -> AbstractMatrix{T}

Compute signatures for a batch of paths on GPU using KernelAbstractions.

This function provides automatic GPU acceleration when the input is a GPU array
(CuArray, ROCArray, MtlArray, etc.). It works with any KernelAbstractions-compatible
backend.

**Current Implementation**: Uses work-sharing parallelization where GPU threads
are assigned paths dynamically. Each thread processes complete paths sequentially.
Future versions will optimize the inner Horner loops for GPU.

# Arguments
- `paths::AbstractArray{T,3}`: Input paths (N × D × B) on GPU/CPU
  - `N ≥ 2`: number of time points per path
  - `D ≥ 1`: spatial dimension
  - `B ≥ 1`: batch size (number of paths)
- `m::Int`: Truncation level (`m ≥ 1`)
- `backend`: KernelAbstractions backend (auto-detected from array type)

# Returns
- `AbstractMatrix{T}`: Signature matrix (sig_len × B) on same device as input
  where `sig_len = D + D² + ... + D^m`

# Example
```julia
using CUDA
using ChenSignatures

# Move batch to GPU
paths_gpu = CuArray(randn(Float32, 50, 2, 10000))  # 10K paths

# Compute signatures on GPU (returns CuMatrix)
sigs_gpu = ChenSignatures.sig_batch_gpu(paths_gpu, 4)

# Transfer result to CPU
sigs_cpu = Array(sigs_gpu)
```

# Performance Notes
- Current version processes paths in parallel but uses sequential Horner updates
- Expected speedup: 5-15x vs single-threaded CPU for large batches (>1000 paths)
- Future optimizations will target 20-50x speedup by parallelizing inner loops

See also: [`sig`](@ref)
"""
function sig_batch_gpu(
    paths::AbstractArray{T,3},
    m::Int;
    backend = KernelAbstractions.get_backend(paths)
) where T
    N, D, B = size(paths)

    # Validation
    N >= 2 || throw(ArgumentError("Paths must have at least 2 points, got N=$N"))
    D >= 1 || throw(ArgumentError("Path dimension must be at least 1, got D=$D"))
    m >= 1 || throw(ArgumentError("Signature level must be at least 1, got m=$m"))
    B >= 1 || throw(ArgumentError("Batch size must be at least 1, got B=$B"))

    # Compute sizes and offsets
    offsets_cpu = level_starts0(D, m)
    offsets = KernelAbstractions.allocate(backend, Int, length(offsets_cpu))
    copyto!(offsets, offsets_cpu)

    tensor_len = offsets_cpu[end]  # With padding
    sig_len = sum(D^k for k in 1:m)  # Without padding
    ws_len = m > 1 ? D^(m-1) : 1

    # Precompute 1/k on CPU and transfer to device
    inv_level_cpu = [one(T) / T(k) for k in 1:m]
    inv_level = KernelAbstractions.allocate(backend, T, m)
    copyto!(inv_level, inv_level_cpu)

    # Allocate output and workspaces (2D arrays for better coalescing)
    results = KernelAbstractions.zeros(backend, T, sig_len, B)
    ws_B1_all = KernelAbstractions.zeros(backend, T, ws_len, B)
    ws_B2_all = KernelAbstractions.zeros(backend, T, ws_len, B)
    tensor_coeffs_all = KernelAbstractions.zeros(backend, T, tensor_len, B)

    # Launch GPU kernel - one thread per path
    kernel! = sig_batch_kernel!(backend)
    event = kernel!(
        results,
        paths,
        offsets,
        ws_B1_all,
        ws_B2_all,
        tensor_coeffs_all,
        inv_level,
        Val(D),
        Val(m);
        ndrange = B
    )

    # Wait for completion (some backends return nothing)
    if !isnothing(event)
        wait(event)
    end

    return results
end

# ============================================================================
# Helper: Detect if running on GPU
# ============================================================================

"""
    is_gpu_array(x)

Check if an array is on GPU (not CPU Vector/Matrix).
"""
is_gpu_array(x::Array) = false
is_gpu_array(x::AbstractArray) = true  # Assume non-Array types are GPU
