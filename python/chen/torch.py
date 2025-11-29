"""PyTorch autograd wrapper for chen.sig using Julia's Zygote/ChainRules rrule"""

# CRITICAL: Import juliacall BEFORE torch to avoid segfaults
from juliacall import Main as jl
import numpy as np
import torch

# Load Zygote for gradients (chen module already loads ChenSignatures)
jl.seval("using Zygote")


class SigFunction(torch.autograd.Function):
    """
    Custom autograd function that uses Julia's Zygote/ChainRules rrule
    """
    
    @staticmethod
    def forward(ctx, path, m):
        """
        Forward pass: compute signature
        
        Args:
            path: torch tensor (N, d)
            m: int, truncation level
        """
        # Import chen here to avoid circular imports
        import chen
        
        # Convert to numpy
        path_np = path.detach().cpu().numpy()
        
        # Use existing chen.sig function
        sig_np = chen.sig(path_np, m)
        
        # Save for backward
        ctx.save_for_backward(path)
        ctx.m = m
        
        # Convert back to torch
        return torch.from_numpy(sig_np).to(path.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient using Julia's Zygote (which uses our rrule)
        
        Args:
            grad_output: gradient of loss w.r.t. signature output
        """
        path, = ctx.saved_tensors
        m = ctx.m
        
        # Convert to numpy
        path_np = path.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()
        
        # Use Zygote to compute gradient - it will automatically use our rrule!
        grad_path = jl.seval("""
        function(path, m, grad_output)
            # Zygote will use the rrule we defined for sig
            loss(p) = sum(ChenSignatures.sig(p, m) .* grad_output)
            grad_tuple = Zygote.gradient(loss, path)
            return grad_tuple[1]
        end
        """)(path_np, m, grad_output_np)
        
        grad_path_np = np.asarray(grad_path)
        
        # Convert back to torch
        grad_path_torch = torch.from_numpy(grad_path_np).to(path.device)
        
        # Return gradients (None for m since it's not differentiable)
        return grad_path_torch, None


def sig_torch(path, m):
    """
    Compute signature with PyTorch autograd support.
    
    Uses Julia's Zygote for automatic differentiation, which in turn
    uses the ChainRules rrule defined for sig.
    
    Args:
        path: torch.Tensor of shape (N, d)
        m: int, truncation level
        
    Returns:
        torch.Tensor of signature coefficients
        
    Example:
        >>> # IMPORTANT: Import chen.torch before importing torch in your script!
        >>> from chen.torch import sig_torch
        >>> import torch  # Import torch AFTER chen.torch
        >>> 
        >>> path = torch.randn(100, 5, requires_grad=True)
        >>> sig = sig_torch(path, m=3)
        >>> loss = sig.sum()
        >>> loss.backward()
        >>> print(path.grad.shape)  # (100, 5)
    """
    return SigFunction.apply(path, m)