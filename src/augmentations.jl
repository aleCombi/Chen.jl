"""
    time_augment(path; Tspan=one(eltype(path)), times=nothing)

Add a monotonically increasing time coordinate as the first column/entry of a path.

Supported inputs:
- `AbstractMatrix{T}` with shape `(N, D)` → `(N, D+1)`
- `AbstractArray{T,3}` with shape `(N, D, B)` → `(N, D+1, B)`
- `AbstractVector{SVector{D,T}}` → `Vector{SVector{D+1,T}}`

Arguments
- `Tspan`: total time horizon (default `one(T)`); ignored when `times` is provided.
- `times`: optional length-`N` vector of time stamps (must be convertible to `T`).

Notes
- The time column is placed first to match common signature practice.
- `times` is not strictly validated for monotonicity, but non-monotone grids will
  generally produce uninterpretable signatures.
"""
function time_augment(path::AbstractMatrix{T}; Tspan=one(T), times=nothing) where {T}
    N, D = size(path)
    N >= 1 || throw(ArgumentError("Path must have at least 1 point, got N=$N"))
    D >= 1 || throw(ArgumentError("Path dimension must be at least 1, got D=$D"))

    tgrid = _time_grid(T, N, Tspan, times)
    out = Array{T}(undef, N, D + 1)
    @views begin
        out[:, 1] .= tgrid
        out[:, 2:end] .= path
    end
    return out
end

function time_augment(paths::AbstractArray{T,3}; Tspan=one(T), times=nothing) where {T}
    N, D, B = size(paths)
    N >= 1 || throw(ArgumentError("Paths must have at least 1 point, got N=$N"))
    D >= 1 || throw(ArgumentError("Path dimension must be at least 1, got D=$D"))
    B >= 1 || throw(ArgumentError("Batch size must be at least 1, got B=$B"))

    tgrid = _time_grid(T, N, Tspan, times)
    out = Array{T}(undef, N, D + 1, B)
    @inbounds for b in 1:B
        @views begin
            out[:, 1, b] .= tgrid
            out[:, 2:end, b] .= paths[:, :, b]
        end
    end
    return out
end

function time_augment(path::AbstractVector{SVector{D,T}}; Tspan=one(T), times=nothing) where {D,T}
    N = length(path)
    N >= 1 || throw(ArgumentError("Path must have at least 1 point, got N=$N"))

    tgrid = _time_grid(T, N, Tspan, times)
    out = Vector{SVector{D + 1, T}}(undef, N)
    @inbounds for i in 1:N
        out[i] = SVector{D + 1, T}(tgrid[i], path[i]...)
    end
    return out
end

"""
    lead_lag(path)

Lead–lag transform of a path. Maps `(N, D)` points to `(2N-1, 2D)` points using
the standard duplication scheme:

    (x_i, x_i), (x_i, x_{i+1}), (x_{i+1}, x_{i+1}), ...

Supported inputs:
- `AbstractMatrix{T}` with shape `(N, D)` → `(2N-1, 2D)`
- `AbstractArray{T,3}` with shape `(N, D, B)` → `(2N-1, 2D, B)`
- `AbstractVector{SVector{D,T}}` → `Vector{SVector{2D,T}}`
"""
function lead_lag(path::AbstractMatrix{T}) where {T}
    N, D = size(path)
    N >= 1 || throw(ArgumentError("Path must have at least 1 point, got N=$N"))
    D >= 1 || throw(ArgumentError("Path dimension must be at least 1, got D=$D"))

    out = Array{T}(undef, 2N - 1, 2D)
    idx = 1
    @inbounds for i in 1:(N-1)
        @views xi = path[i, :]
        @views xip1 = path[i + 1, :]

        @views begin
            out[idx, 1:D] .= xi
            out[idx, D+1:end] .= xi
        end
        idx += 1

        @views begin
            out[idx, 1:D] .= xi
            out[idx, D+1:end] .= xip1
        end
        idx += 1
    end

    @views begin
        xn = path[N, :]
        out[idx, 1:D] .= xn
        out[idx, D+1:end] .= xn
    end

    return out
end

function lead_lag(paths::AbstractArray{T,3}) where {T}
    N, D, B = size(paths)
    N >= 1 || throw(ArgumentError("Paths must have at least 1 point, got N=$N"))
    D >= 1 || throw(ArgumentError("Path dimension must be at least 1, got D=$D"))
    B >= 1 || throw(ArgumentError("Batch size must be at least 1, got B=$B"))

    out = Array{T}(undef, 2N - 1, 2D, B)
    @inbounds for b in 1:B
        idx = 1
        for i in 1:(N-1)
            @views xi = paths[i, :, b]
            @views xip1 = paths[i + 1, :, b]

            @views begin
                out[idx, 1:D, b] .= xi
                out[idx, D+1:end, b] .= xi
            end
            idx += 1

            @views begin
                out[idx, 1:D, b] .= xi
                out[idx, D+1:end, b] .= xip1
            end
            idx += 1
        end

        @views begin
            xn = paths[N, :, b]
            out[idx, 1:D, b] .= xn
            out[idx, D+1:end, b] .= xn
        end
    end

    return out
end

function lead_lag(path::AbstractVector{SVector{D,T}}) where {D,T}
    N = length(path)
    N >= 1 || throw(ArgumentError("Path must have at least 1 point, got N=$N"))

    out = Vector{SVector{2D, T}}(undef, 2N - 1)
    idx = 1
    @inbounds for i in 1:(N-1)
        xi = path[i]
        xip1 = path[i + 1]

        out[idx] = _lead_lag_point(xi, xi)
        idx += 1

        out[idx] = _lead_lag_point(xi, xip1)
        idx += 1
    end

    out[idx] = _lead_lag_point(path[N], path[N])
    return out
end

# Convenience wrappers that apply augmentations before signature computation
"""
    sig_time(path, m::Int; kwargs...)

Compute the truncated signature of a time-augmented path.

Applies [`time_augment`](@ref) to prepend a time coordinate, then delegates to [`sig`](@ref).
Keyword arguments are forwarded to `time_augment` (e.g. `Tspan`, `times`).
"""
function sig_time(path, m::Int; kwargs...)
    return sig(time_augment(path; kwargs...), m)
end

"""
    sig_leadlag(path, m::Int)

Compute the truncated signature of a path after applying the lead-lag transform.

Calls [`lead_lag`](@ref) to duplicate coordinates using the standard scheme and then
computes [`sig`](@ref) on the transformed path.
"""
function sig_leadlag(path, m::Int)
    return sig(lead_lag(path), m)
end

"""
    logsig_time(path, basis::BasisCache; kwargs...)

Log-signature of a time-augmented path using a precomputed basis.

Validates that the augmented dimension matches `basis.d`, applies [`time_augment`](@ref)
with the provided keyword arguments, and then computes [`logsig`](@ref).
"""
function logsig_time(path, basis::BasisCache; kwargs...)
    aug_dim = _augmented_dim(path; kind=:time)
    basis.d == aug_dim || throw(ArgumentError(
        "Basis dimension $(basis.d) does not match time-augmented path dimension $aug_dim"
    ))
    return logsig(time_augment(path; kwargs...), basis)
end

"""
    logsig_leadlag(path, basis::BasisCache)

Log-signature of a lead-lag transformed path using a precomputed basis.

Checks that the lead-lag dimension matches `basis.d`, applies [`lead_lag`](@ref), and
computes [`logsig`](@ref) on the augmented path.
"""
function logsig_leadlag(path, basis::BasisCache)
    aug_dim = _augmented_dim(path; kind=:leadlag)
    basis.d == aug_dim || throw(ArgumentError(
        "Basis dimension $(basis.d) does not match lead-lag path dimension $aug_dim"
    ))
    return logsig(lead_lag(path), basis)
end

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
function _time_grid(::Type{T}, N::Int, Tspan, times) where {T}
    if times === nothing
        return range(zero(T), stop=convert(T, Tspan), length=N)
    else
        length(times) == N || throw(ArgumentError(
            "times vector length ($(length(times))) must match path length N=$N"
        ))
        return T.(times)
    end
end

@inline function _lead_lag_point(x::SVector{D,T}, y::SVector{D,T}) where {D,T}
    return SVector{2D, T}(x..., y...)
end

function _augmented_dim(path::AbstractMatrix; kind)
    D = size(path, 2)
    return kind === :time ? D + 1 : 2 * D
end
function _augmented_dim(path::AbstractArray{<:Any,3}; kind)
    D = size(path, 2)
    return kind === :time ? D + 1 : 2 * D
end
function _augmented_dim(path::AbstractVector{<:SVector{D}}; kind) where {D}
    return kind === :time ? D + 1 : 2 * D
end
