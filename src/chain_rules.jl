function ChainRulesCore.rrule(::typeof(sig), path::AbstractMatrix{T}, m::Int) where T
    # 1. Compute Primal (Forward pass)
    result = sig(path, m)
    
    # 2. Define Pullback (Backward pass)
    function sig_pullback(ȳ_raw)
        # Unthunk retrieves the gradient from ChainRules wrapper
        ȳ = ChainRulesCore.unthunk(ȳ_raw)
        
        # -----------------------------------------------------------
        # OPTIMIZATION 1: Ensure Memory Layout
        # -----------------------------------------------------------
        # PyTorch sends a 'PyArray'. Enzyme requires standard 'Array'.
        # If it's already an Array (e.g. from Julia), use it. 
        # Otherwise copy it to a standard Julia Matrix.
        path_dense = path isa Array ? path : copy(path)
        
        # Allocate gradient buffer (Shadow)
        grad = zeros(T, size(path_dense))
        
        # -----------------------------------------------------------
        # OPTIMIZATION 2: Avoid Closures for Enzyme
        # -----------------------------------------------------------
        # We define a helper that takes ALL arguments explicitly.
        # This prevents "boxing" of 'm' or 'ȳ' which confuses Enzyme.
        function signature_loss(p, _m, _y)
            return dot(sig(p, _m), _y)
        end

        # -----------------------------------------------------------
        # EXECUTION: Enzyme Autodiff
        # -----------------------------------------------------------
        # We allow Enzyme to differentiate 'signature_loss'.
        # - Active: The return value (scalar dot product) has a derivative.
        # - Duplicated(path, grad): Differentiate wrt 'path', store in 'grad'.
        # - Const(m): 'm' is an integer constant.
        # - Const(ȳ): 'ȳ' is the incoming gradient, treated as constant data here.
        Enzyme.autodiff(Reverse, signature_loss, Active, 
                        Duplicated(path_dense, grad), 
                        Const(m), 
                        Const(ȳ))
        
        return (NoTangent(), grad, NoTangent())
    end
    
    return result, sig_pullback
end