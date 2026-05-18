import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # @torch.compile
    def forward(self, x):
        # 1. Keep original dtype for the final multiply
        input_dtype = x.dtype
        
        # 2. Perform variance math in float32 to avoid overflow/underflow
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        
        # 3. Apply rsqrt and scale
        # Using (variance + self.eps).rsqrt() is the standard pattern
        norm_x = x_f32 * torch.rsqrt(variance + self.eps)
        
        # 4. Cast back and apply learnable weight
        return (norm_x.to(input_dtype)) * self.weight
