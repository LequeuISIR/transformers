import torch
import torch.nn as nn
import torch.nn.functional as F

class PosBERTSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, sem_size=None, pos_size=None, separate_w3=False, multiple_of=8):
        super().__init__()
        self.separate_w3 = separate_w3
        
        # Fused W1 and W2
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=False)
        
        if not self.separate_w3:
            self.w3 = nn.Linear(hidden_features, in_features, bias=False)
        else:
            assert sem_size and pos_size
            total_in = sem_size + pos_size
            
            # Calculate and round pos_h to maintain GPU alignment (multiple of 8)
            pos_h_raw = int(hidden_features * (pos_size / total_in))
            self.pos_h = (pos_h_raw // multiple_of) * multiple_of
            self.sem_h = hidden_features - self.pos_h
            
            self.w3 = nn.Linear(hidden_features, in_features, bias=False)
            
            # 1. Create the mask
            mask = self._create_block_mask(pos_size, sem_size, self.pos_h, self.sem_h)
            self.register_buffer('mask', mask)
            
            # 2. Apply the mask to the weights immediately
            with torch.no_grad():
                self.w3.weight.mul_(self.mask)

    def _create_block_mask(self, pos_size, sem_size, pos_h, sem_h):
        mask = torch.zeros(pos_size + sem_size, pos_h + sem_h)
        mask[:pos_size, :pos_h] = 1.0     # Position block
        mask[pos_size:, pos_h:] = 1.0     # Semantic block
        return mask

    def forward(self, x):
        combined = self.w12(x)
        gate, value = torch.chunk(combined, 2, dim=-1)
        swiglu_out = F.silu(gate) * value
        
        if not self.separate_w3:
            return self.w3(swiglu_out)
        else:
            # Masking here ensures zero-blocks stay zero during training
            return F.linear(swiglu_out, self.w3.weight * self.mask)