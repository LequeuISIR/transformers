import torch
from torch import nn

class CLSSEPAttentionReplacer(nn.Module):
    def __init__(self, num_heads: int, init_value: float = 0.05):
        super().__init__()
        self.num_heads = num_heads
        # Initialize as parameters
        self.thetas = nn.Parameter(torch.full((4, num_heads), init_value))

    def forward(self, attn: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        B, H, L, _ = attn.shape
        device = attn.device
        
        # 1. Get indices (This logic remains necessary but is fast)
        # Note: argmax is fine, but ensure pad_mask is [B, L] or [B, 1, L] 
        # to avoid redundant [B, H] index calculations if heads are the same.
        row_mask = pad_mask.any(dim=-1) 
        cls_idx = row_mask.float().argmax(dim=-1)
        sep_idx = L - 1 - row_mask.flip(dims=[-1]).float().argmax(dim=-1)

        # 2. Create Coordinate Grids
        # We want to find where (i == cls) or (j == cls), etc.
        range_l = torch.arange(L, device=device)
        
        # [B, H, L]
        is_cls_row = (range_l == cls_idx.unsqueeze(-1))
        is_sep_row = (range_l == sep_idx.unsqueeze(-1))
        is_cls_col = is_cls_row # same indices for columns in square attn
        is_sep_col = is_sep_row

        # 3. Apply logic via where/masking
        # This is more "functional" and much easier for torch.compile to optimize
        # We prioritize columns over rows as per your requirement
        
        # Row values
        out = torch.where(is_cls_row.unsqueeze(-1), self.thetas[0].view(1, H, 1, 1), attn)
        out = torch.where(is_sep_row.unsqueeze(-1), self.thetas[1].view(1, H, 1, 1), out)
        
        # Column values (overwriting rows)
        out = torch.where(is_cls_col.unsqueeze(-2), self.thetas[2].view(1, H, 1, 1), out)
        out = torch.where(is_sep_col.unsqueeze(-2), self.thetas[3].view(1, H, 1, 1), out)

        return out


class OldCLSSEPAttentionReplacer(nn.Module):
    """
    Fully vectorized attention replacer for [CLS] and [SEP] with head-specific learnable scalars.
    Column replacements overwrite row replacements at intersections.
    """
    def __init__(self, num_heads: int, init_value: float = 0.05):
        super().__init__()
        self.num_heads = num_heads
        self.register_parameter("theta_cls_out", nn.Parameter(torch.full((num_heads,), init_value)))
        self.register_parameter("theta_cls_in",  nn.Parameter(torch.full((num_heads,), init_value)))
        self.register_parameter("theta_sep_out", nn.Parameter(torch.full((num_heads,), init_value)))
        self.register_parameter("theta_sep_in",  nn.Parameter(torch.full((num_heads,), init_value)))

    def forward(self, attn: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn: Tensor [B, H, L, L]
            pad_mask: BoolTensor [B, H, L, L] (True = non-pad token)
        Returns:
            Tensor [B, H, L, L] with replaced entries.
        """
        B, H, L, _ = attn.shape
        device = attn.device
        dtype = attn.dtype


        cls_idx = pad_mask.any(dim=-1).float().argmax(dim=-1)  # [B,H]
        sep_idx = (pad_mask.any(dim=-1).flip(dims=[-1]).float().argmax(dim=-1))
        sep_idx = L - 1 - sep_idx                                # [B,H] # [B]

        # Copy attention to avoid in-place issues
        out = attn.clone()

        batch_idx = torch.arange(B, device=device)[:, None]  # [B,1]
        head_idx  = torch.arange(H, device=device)[None, :]  # [1,H]

        # Expand theta to [B,H,L] for broadcasting
        theta_cls_out_exp = self.theta_cls_out.view(1,H,1).expand(B,H,L)
        theta_sep_out_exp = self.theta_sep_out.view(1,H,1).expand(B,H,L)
        theta_cls_in_exp  = self.theta_cls_in.view(1,H,1).expand(B,H,L)
        theta_sep_in_exp  = self.theta_sep_in.view(1,H,1).expand(B,H,L)

        # Replace rows
        out[batch_idx, head_idx, cls_idx, :] = theta_cls_out_exp.to(out.dtype)
        out[batch_idx, head_idx, sep_idx, :] = theta_sep_out_exp.to(out.dtype)

        # Replace columns
        out[batch_idx, head_idx, :, cls_idx] = theta_cls_in_exp.to(out.dtype)
        out[batch_idx, head_idx, :, sep_idx] = theta_sep_in_exp.to(out.dtype)

        # # --- Replace CLS rows (source -> others) ---
        # theta_cls_out_exp = self.theta_cls_out.view(1,H,1).expand(B,H,L)
        # print('theta_cls_out_exp', theta_cls_out_exp.shape)
        # out[batch_idx[:, None], torch.arange(H, device=device)[None,:], cls_idx[:, None], :] = theta_cls_out_exp


        # # --- Replace SEP rows ---
        # theta_sep_out_exp = self.theta_sep_out.view(1,H,1).expand(B,H,L)
        # out[batch_idx[:, None], torch.arange(H)[None,:], sep_idx[:, None], :] = theta_sep_out_exp

        # # --- Replace CLS columns (target <- others) ---
        # theta_cls_in_exp = self.theta_cls_in.view(1,H,1).expand(B,H,L)
        # out[batch_idx[:, None], torch.arange(H)[None,:], :, cls_idx[:, None]] = theta_cls_in_exp

        # # --- Replace SEP columns ---
        # theta_sep_in_exp = self.theta_sep_in.view(1,H,1).expand(B,H,L)
        # out[batch_idx[:, None], torch.arange(H)[None,:], :, sep_idx[:, None]] = theta_sep_in_exp

        return out

if __name__ == "__main__" :
    # Setup: 1 Batch, 1 Head, 3x3 Matrix
    B, H, L = 1, 1, 8
    replacer = CLSSEPAttentionReplacer(num_heads=H, init_value=0.05)
    
    # Manually set parameters to distinguishable values
    with torch.no_grad():
        replacer.thetas[0].fill_(1.1) # Row CLS
        replacer.thetas[1].fill_(2.2) # Row SEP
        replacer.thetas[2].fill_(8.8)  # Col CLS (Priority)
        replacer.thetas[3].fill_(9.9)  # Col SEP (Priority)

    # Mock attention matrix (all zeros)
    attn = torch.ones((B, H, L, L))
    
    # Mock pad_mask: [B, L] -> True means not a pad
    # Indices: 0=CLS, 1=Data, 2=SEP
    pad_mask = torch.tensor([[True for i in range(L)]]) 
    # Expand to match expected [B, H, L, L] if your code requires it, 
    # though usually mask logic uses [B, L]
    pad_mask_expanded = pad_mask.view(B, 1, L).expand(B, H, L)

    output = replacer(attn, pad_mask_expanded)

    print("--- Resulting Matrix (Head 0) ---")
    print(output[0, 0])

    # Validation Logic
    expected_cls_val = 8.8
    actual_intersection = output[0, 0, 0, 0].item()
    
    print("\n--- Validation ---")
    if torch.allclose(output[0, 0, 0, :], torch.tensor([8.8, 1.1, 1.1])):
        print("✅ Row replacement working (with Col priority at 0,0)")
    else:
        print("❌ Row replacement failed")

    if actual_intersection == expected_cls_val:
        print(f"✅ Priority Check: Column (8.8) correctly overwrote Row (1.1) at [0,0]")
    else:
        print(f"❌ Priority Check: Expected 8.8 at [0,0], got {actual_intersection}")