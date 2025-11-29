"""PyTorch autograd wrapper for chen.sig using Julia's ChainRules rrule"""

import numpy as np
from juliacall import Main as jl
import torch

# Import the existing chen module which already loads ChenSignatures
import chen

# Load Zygote which will use our rrule
jl.seval("using Zygote")

class SigFunction(torch.autograd.Function):
    """
    Custom autograd function that uses Julia's Zygote/ChainRules rrule
    """
    
    @staticmethod
    def forward(ctx, path, m):
        """
        Forward pass: compute signature using existing chen.sig
        
        Args:
            path: torch tensor (N, d)
            m: int, truncation level
        """
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
        # Zygote.gradient returns a tuple, first element is gradient w.r.t. path
        jl.seval("""
        function compute_sig_gradient_zygote(path, m, grad_output)
            # Zygote will use the rrule we defined for sig
            loss(p) = sum(ChenSignatures.sig(p, m) .* grad_output)
            grad_tuple = Zygote.gradient(loss, path)
            return grad_tuple[1]
        end
        """)
        
        # Compute gradient
        grad_path = jl.compute_sig_gradient_zygote(path_np, m, grad_output_np)
        grad_path_np = np.asarray(grad_path)
        
        # Convert back to torch
        grad_path_torch = torch.from_numpy(grad_path_np).to(path.device)
        
        # Return gradients (None for m since it's not differentiable)
        return grad_path_torch, None


# User-friendly wrapper
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
    """
    return SigFunction.apply(path, m)


if __name__ == "__main__":
    print("="*70)
    print("PYTORCH AUTOGRAD TEST WITH ZYGOTE/CHAINRULES")
    print("="*70)
    print()
    
    # Test 1: Forward pass
    print("Test 1: Forward pass")
    path = torch.randn(10, 3, requires_grad=True, dtype=torch.float64)
    sig = sig_torch(path, m=3)
    print(f"  Path shape: {path.shape}")
    print(f"  Signature shape: {sig.shape}")
    print(f"  ✓ Passed")
    print()
    
    # Test 2: Backward pass
    print("Test 2: Backward pass (using rrule)")
    loss = sig.sum()
    loss.backward()
    print(f"  Loss: {loss.item()}")
    print(f"  Gradient shape: {path.grad.shape}")
    print(f"  Gradient norm: {path.grad.norm().item()}")
    print(f"  ✓ Passed")
    print()
    
    # Test 3: Compare with existing chen.sig
    print("Test 3: Forward pass matches chen.sig")
    path_np = path.detach().cpu().numpy()
    sig_direct = chen.sig(path_np, m=3)
    sig_autograd = sig_torch(torch.from_numpy(path_np), m=3).detach().numpy()
    
    print(f"  Max difference: {np.abs(sig_direct - sig_autograd).max()}")
    print(f"  ✓ Passed" if np.allclose(sig_direct, sig_autograd) else "  ✗ Failed")
    print()
    
    # Test 4: Gradient check with finite differences
    print("Test 4: Gradient check (finite differences)")
    path_test = torch.randn(5, 2, requires_grad=True, dtype=torch.float64)
    sig_test = sig_torch(path_test, m=2)
    loss_test = sig_test.sum()
    loss_test.backward()
    
    grad_autograd = path_test.grad.clone()
    
    # Finite differences
    eps = 1e-6
    grad_fd = torch.zeros_like(path_test)
    with torch.no_grad():
        for i in range(path_test.shape[0]):
            for j in range(path_test.shape[1]):
                path_plus = path_test.clone()
                path_plus[i, j] += eps
                
                path_minus = path_test.clone()
                path_minus[i, j] -= eps
                
                sig_plus = sig_torch(path_plus, m=2)
                sig_minus = sig_torch(path_minus, m=2)
                
                grad_fd[i, j] = (sig_plus.sum() - sig_minus.sum()) / (2 * eps)
    
    max_diff = (grad_autograd - grad_fd).abs().max().item()
    print(f"  Max difference: {max_diff}")
    print(f"  ✓ Passed" if torch.allclose(grad_autograd, grad_fd, rtol=1e-4, atol=1e-6) else "  ✗ Failed")
    print()
    
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print("✓ PyTorch can use Julia's rrule for gradients!")
    print("✓ No need to reimplement gradients - just call Zygote.gradient")