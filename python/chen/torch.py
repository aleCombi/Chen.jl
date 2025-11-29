"""PyTorch autograd wrapper for chen.sig using Julia's Zygote/ChainRules rrule"""

# CRITICAL: Import juliacall BEFORE torch to avoid segfaults
from juliacall import Main as jl
import numpy as np
import torch

# Load Zygote
jl.seval("using Zygote")
jl.seval("using ChenSignatures")

# Define the helper function ONCE in Julia.
# This returns a tuple: (result, pullback_function)
jl.seval("""
function _sig_forward_backward(path, m)
    # Zygote.pullback returns (y, back)
    # y is the result, back is a function: dy -> (dx,)
    return Zygote.pullback(p -> ChenSignatures.sig(p, m), path)
end
""")

# Pre-fetch the function handle
_sig_pullback_fn = jl.seval("_sig_forward_backward")

class SigFunction(torch.autograd.Function):
    """
    Optimized autograd function that caches the Zygote pullback closure.
    """
    
    @staticmethod
    def forward(ctx, path, m):
        """
        Forward pass: compute signature and keep Zygote pullback alive.
        """
        # Ensure contiguous Float64 array for Julia
        path_np = np.ascontiguousarray(path.detach().cpu().numpy(), dtype=np.float64)
        
        # 1. Call Zygote.pullback immediately.
        res_jl, back_jl = _sig_pullback_fn(path_np, m)
        
        # 2. Save the Julia 'back' closure in the context.
        ctx.back_jl = back_jl
        ctx.device = path.device
        
        # 3. Return result as torch tensor
        return torch.from_numpy(np.array(res_jl)).to(path.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Invoke the pre-compiled Julia pullback.
        """
        # Convert gradient to numpy
        grad_output_np = np.ascontiguousarray(grad_output.detach().cpu().numpy(), dtype=np.float64)
        
        # 1. Retrieve the Julia pullback closure
        back_jl = ctx.back_jl
        
        # 2. Call it. 
        # back(dy) -> (d_path,)
        grads_tuple = back_jl(grad_output_np)
        
        # 3. Extract the gradient for 'path'
        grad_path_jl = grads_tuple[0]
        
        # 4. Convert back to torch
        grad_path_torch = torch.from_numpy(np.array(grad_path_jl)).to(ctx.device)
        
        # Return gradients (None for m)
        return grad_path_torch, None


def sig_torch(path, m):
    """
    Compute signature with PyTorch autograd support.
    
    Args:
        path: torch.Tensor of shape (N, d)
        m: int, truncation level
    """
    return SigFunction.apply(path, m)