import torch
import torch.nn as nn
import math

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance):
        raise NotImplementedError
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        # We need (2 * max_distance - 1) slots to cover negative and positive offsets
        self.bias_table = nn.Parameter(
            torch.zeros(num_heads, 2 * max_distance)
        )

    def forward(self, seq_len):
        # 1. Create a matrix of relative distances
        # range(seq_len) looks like, [0, 1, 2]
        pos1 = torch.arange(seq_len, dtype=torch.long, device=self.bias_table.device).view(-1, 1)
        pos2 = torch.arange(seq_len, dtype=torch.long, device=self.bias_table.device).view(1, -1)
        # diffs[i, j] = i - j
        relative_indices = pos2 - pos1
        
        # 2. Shift indices to be non-negative (from 0 to 2*L - 2)
        relative_indices = relative_indices + (self.max_distance)
        # 3. Index into the bias table 
        # Output shape: (num_heads, seq_len, seq_len)
        return self.bias_table[:, relative_indices]
    


class RelativePositionBucketedBias(nn.Module):
    def __init__(self, num_heads, max_seq_len, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Learnable parameters
        self.relative_attention_bias = nn.Parameter(
            torch.zeros(num_buckets, num_heads)
        )

        # PRE-COMPUTE bucket indices once
        grid_q = torch.arange(max_seq_len, dtype=torch.long).view(-1, 1)
        grid_k = torch.arange(max_seq_len, dtype=torch.long).view(1, -1)
        relative_position = grid_k - grid_q
        
        indices = self._relative_position_bucket(
            relative_position, num_buckets=self.num_buckets, max_distance=self.max_distance
        )

        # Register as a buffer so it moves with the model to GPU
        self.register_buffer("bucket_indices", indices, persistent=True)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """
        Maps relative distances to bucket indices.
        """
        relative_buckets = 0
        # For bidirectional: use half buckets for negative, half for positive
        # For causal: you'd only care about one direction. 
        # Here we assume bidirectional/symmetric:
        n = -relative_position
        
        num_buckets //= 2
        relative_buckets += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        # Half of the buckets are for 'exact' near distances
        max_exact = num_buckets // 2
        is_small = n < max_exact
        

        # The other half are for logarithmic 'far' distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / 
            math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, n, val_if_large)
        
        return relative_buckets

    
    def forward(self, seq_len):
        # 1. Slice pre-computed buffer
        indices = self.bucket_indices[:seq_len, :seq_len]

        # 2. Gather biases [num_heads, seq_len, seq_len]
        bias_table = self.relative_attention_bias.T 
        out = bias_table[:, indices] 
        
        # 3. Zero out boundaries
        # We use [:, 0, :] for the first row and [:, :, 0] for the first column
        # This handles all heads simultaneously.
        out[:, 0, :] = 0  # First row
        out[:, -1, :] = 0 # Last row
        out[:, :, 0] = 0  # First column
        out[:, :, -1] = 0 # Last column (fixed your ':: -1' typo)

        return out.unsqueeze(0) # Final shape: [1, H, L, L]
    


import torch
from torch.nn import Parameter

import torch
from torch.nn import Parameter

class DoubleKerpleLog(torch.nn.Module):
    """
    Asymmetric Kernelized T5 Relative Position Bias.
    Learns different p and a parameters for 'past' vs 'future' tokens.
    """

    def __init__(self, num_attention_heads):
        super().__init__()
        self.heads = num_attention_heads
        self.eps = 1e-2
        
        
        # Parameters for tokens behind (past/backward)
        self.bias_p_past = Parameter(torch.rand(self.heads, 1, 1, dtype=torch.long) * 2)
        self.bias_a_past = Parameter(torch.rand(self.heads, 1, 1, dtype=torch.long) * 1)
        
        # Parameters for tokens in front (future/forward)
        self.bias_p_future = Parameter(torch.rand(self.heads, 1, 1, dtype=torch.long) * 2)
        self.bias_a_future = Parameter(torch.rand(self.heads, 1, 1, dtype=torch.long) * 1)

        self.register_buffer("cached_matrix", None, persistent=False)
        self.cached_seq_len = None

    def forward(self, x):
        # x shape: [batch, heads, seq_q, seq_k]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        
        if self.cached_seq_len != seq_len_k:
            # Standard signed distance matrix (m - n)
            # Positive values = tokens in the past (relative to query)
            # Negative values = tokens in the future (relative to query)
            grid = torch.arange(seq_len_k, device=x.device)
            dist = grid.view(-1, 1) - grid.view(1, -1) 
            
            self.cached_seq_len = seq_len_k
            self.cached_matrix = dist.to(x.dtype)
        
        dist = self.cached_matrix
        
        # Create masks for past and future
        # dist > 0: current token is at index i, looking at index j where i > j (past)
        # dist < 0: current token is at index i, looking at index j where i < j (future)
        mask_past = (dist > 0).to(x.dtype)
        mask_future = (dist < 0).to(x.dtype)
        
        # Clamp parameters for stability
        p_p, a_p = self.bias_p_past.clamp(min=self.eps), self.bias_a_past.clamp(min=self.eps)
        p_f, a_f = self.bias_p_future.clamp(min=self.eps), self.bias_a_future.clamp(min=self.eps)
        
        # Calculate kernels separately
        # We use torch.abs(dist) for both to ensure log(1 + a*|dist|) is valid
        abs_dist = torch.abs(dist)
        
        bias_past = -p_p * torch.log(1 + a_p * abs_dist) * mask_past
        bias_future = -p_f * torch.log(1 + a_f * abs_dist) * mask_future
        
        # Combine (diagonal is 0 because both masks are False at dist == 0)
        total_bias = bias_past + bias_future
        
        # Handle slicing for inference if seq_q != seq_k
        if seq_len_q != seq_len_k:
            total_bias = total_bias[-seq_len_q:, :]

        return x + total_bias
    
if __name__ == "__main__" :
    # pos_bias = RelativePositionBias(1, 8)
    # fw = pos_bias(10)
    # print(fw)

    pos_bias = RelativePositionBucketedBias(1, 30, 10, 128)
    fw = pos_bias(30)
    # print(fw)
    
    fw = pos_bias(128)
    print(fw.shape)
    
