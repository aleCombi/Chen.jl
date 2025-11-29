function ChainRulesCore.rrule(::typeof(sig), path::AbstractMatrix{T}, m::Int) where T
    result = sig(path, m)
    
    function sig_pullback(ȳ_raw)
        ȳ = ChainRulesCore.unthunk(ȳ_raw)
        grad = zeros(T, size(path))
        loss(p) = dot(sig(p, m), ȳ)
        autodiff(Reverse, loss, Active, Duplicated(copy(path), grad))
        return (NoTangent(), grad, NoTangent())
    end
    
    return result, sig_pullback
end