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
        # Preserve dtype: convert torch dtype to appropriate numpy dtype
        if path.dtype == torch.float32:
            np_dtype = np.float32
        elif path.dtype == torch.float64:
            np_dtype = np.float64
        else:
            # Default to float64 for other types
            np_dtype = np.float64

        # Ensure contiguous array with preserved dtype for Julia
        path_np = np.ascontiguousarray(path.detach().cpu().numpy(), dtype=np_dtype)

        # 1. Call Zygote.pullback immediately.
        res_jl, back_jl = _sig_pullback_fn(path_np, m)

        # 2. Save the Julia 'back' closure and dtype in the context.
        ctx.back_jl = back_jl
        ctx.device = path.device
        ctx.dtype = path.dtype

        # 3. Return result as torch tensor with original dtype
        return torch.from_numpy(np.array(res_jl)).to(device=path.device, dtype=path.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Invoke the pre-compiled Julia pullback.
        """
        # Preserve dtype from forward pass
        if ctx.dtype == torch.float32:
            np_dtype = np.float32
        elif ctx.dtype == torch.float64:
            np_dtype = np.float64
        else:
            np_dtype = np.float64

        # Convert gradient to numpy with preserved dtype
        grad_output_np = np.ascontiguousarray(grad_output.detach().cpu().numpy(), dtype=np_dtype)

        # 1. Retrieve the Julia pullback closure
        back_jl = ctx.back_jl

        # 2. Call it.
        # back(dy) -> (d_path,)
        grads_tuple = back_jl(grad_output_np)

        # 3. Extract the gradient for 'path'
        grad_path_jl = grads_tuple[0]

        # 4. Convert back to torch with original dtype
        grad_path_torch = torch.from_numpy(np.array(grad_path_jl)).to(device=ctx.device, dtype=ctx.dtype)

        # Return gradients (None for m)
        return grad_path_torch, None


def sig_torch(path, m):
    """
    Compute signature with PyTorch autograd support.

    Args:
        path: torch.Tensor of shape (N, d)
              Supports both float32 and float64 dtypes (preserved throughout)
        m: int, truncation level

    Returns:
        torch.Tensor: Signature of shape (d + d^2 + ... + d^m,)
                     with same dtype and device as input
    """
    return SigFunction.apply(path, m)