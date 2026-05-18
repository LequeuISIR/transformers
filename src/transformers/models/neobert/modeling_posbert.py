
# From https://stackoverflow.com/a/23689767
# From https://github.com/pytorch/pytorch/issues/97899
# From https://github.com/facebookresearch/llama/blob/main/llama/model.py

import math
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# from torch.nn.functional import scaled_dot_product_attention
import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Union
from functools import partial

from xformers.ops import SwiGLU, memory_efficient_attention

from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput
)
# from datasets import Dataset

from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizerFast, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput

from tqdm import tqdm

from .rmsnorm import RMSNorm
from .rotary import precompute_freqs_cis, apply_rotary_emb
from .softpick import softpick
from .override_CLS_SEP import CLSSEPAttentionReplacer
from .relative_position_bias import RelativePositionBucketedBias
from .swiglu import PosBERTSwiGLU

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, attn_activation_fct=torch.softmax, RP_bias = None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    if RP_bias is not None:
        attn_weight += RP_bias

    attn_weight += attn_bias
    attn_weight = attn_activation_fct(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value, attn_weight


def posbert_scaled_dot_product_attention(query, key, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, attention_activation = "softmax") -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
        
    return attn_weight 


class BlockDiagonalLayer(nn.Module):
    def __init__(self, block1_size, block2_size):
        super().__init__()
        # Define the blocks as learnable parameters
        self.block1 = nn.Parameter(torch.randn(block1_size, block1_size))
        self.block2 = nn.Parameter(torch.randn(block2_size, block2_size))

    def forward(self, x):
        # Create the structural block diagonal matrix
        # This happens every forward pass, enforcing the 0-blocks
        W = torch.block_diag(self.block1, self.block2)
        # self.weights = 
        # Linear transformation: y = xW^T (or Wx depending on your shape)
        return torch.matmul(x, W.t())
    

class NeoBERTConfig(PretrainedConfig):
    model_type = "neobert"

    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 768,
        pos_size: int = 384,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        pos_intermediate_size: int | None = None,
        intermediate_size: int | None = None,
        pos_dropout_prob: float =0.1,
        dropout_prob: float =0.1,
        attention_probs_dropout_prob: float =0.1,
        use_only_sem_for_decoding: bool = False,
        mixed_feed_forward: bool = True,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        rms_norm: bool = True,
        rope: bool = True,
        posneobert: bool = False,
        norm_eps: float = 1e-06,
        hidden_act: str = "SwiGLU",
        vocab_size: int = 32064,
        pad_token_id: int = 0,
        max_length: int = 1024,
        flash_attention: bool = True,
        base_scale: float = 1.0 / (960.0**0.5),
        ngpt: bool = False,
        positional_embed_init: str = "random",
        attention_activation: str = "softmax",
        mix_attentions: str = "sum",
        untie_cls: bool = False,
        random_offset = False,
        shared_pos_keys = False,
        relative_pos_bias = False,
        AP_embeddings = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # if rope and posneobert :
        #     raise ValueError("cant be rope and posneobert at the same time")
        if ngpt and posneobert :
            raise NotImplementedError
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")
        if pos_size % num_attention_heads != 0 :
            raise ValueError("Pos size must be divisible by the number of heads.")
        if (not posneobert) and (use_only_sem_for_decoding) :
            raise ValueError("Cannot use RoPE and use only semantic for decoding.")
        if (not posneobert) and (positional_embed_init == "2dim_cosine") :
            raise ValueError("Cannot use RoPE and setup positional embeddings.")
        if (not posneobert) and (shared_pos_keys) :
            raise ValueError("Cannot use RoPE and shared positional embeddings.")
        if (rope) and (relative_pos_bias) :
            raise ValueError("Cannot use RoPE and relative positional bias.")
        # if (rope) and (AP_embeddings) :
        #     raise ValueError("Cannot use RoPE and AP embeddingq.")
        if (not posneobert) and (mix_attentions == "hadamard") :
            raise ValueError("Cannot setup mix attentions with RoPE.")
        
        if posneobert and (not AP_embeddings) :
            raise ValueError("Can't use posneobert without AP embeddings")

        if positional_embed_init not in ["random", "2dim_cosine", "fixed"] :
            raise ValueError
        if attention_activation not in ["softmax", "softpick"] :
            raise ValueError
        if mix_attentions not in ["sum", "hadamard"] :
            raise ValueError

        
        self.hidden_size = hidden_size
        self.pos_size = pos_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        self.dim_head = ((hidden_size + pos_size) // num_attention_heads) if posneobert else hidden_size // num_attention_heads
        self.pos_intermediate_size = pos_intermediate_size or pos_size*4
        self.intermediate_size = intermediate_size or hidden_size*4
        self.pos_dropout_prob = pos_dropout_prob
        self.dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_only_sem_for_decoding = use_only_sem_for_decoding
        self.mixed_feed_forward = mixed_feed_forward
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.rms_norm = rms_norm
        self.rope = rope
        self.posneobert = posneobert
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.flash_attention = flash_attention
        self.base_scale = base_scale
        self.ngpt = ngpt
        self.positional_embed_init = positional_embed_init
        self.attention_activation = attention_activation
        self.untie_cls = untie_cls
        self.mix_attentions = mix_attentions
        self.random_offset = random_offset
        self.shared_pos_keys = shared_pos_keys
        self.relative_pos_bias = relative_pos_bias
        self.AP_embeddings = AP_embeddings
        self.kwargs = kwargs


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        # Attention
        if not self.config.posneobert :
            self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
            self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.resid_dropout = nn.Dropout(config.dropout_prob)
        else :
            if self.config.shared_pos_keys :
                raise NotImplementedError
                self.q_pos = nn.Linear(in_features=config.pos_size, out_features=(config.hidden_size + config.pos_size), bias=False)
            else : 
                self.qkv_pos = nn.Linear(in_features=config.pos_size, out_features=(config.hidden_size + config.pos_size) * 2 + config.pos_size, bias=False)


            self.qkv_sem = nn.Linear(in_features=config.hidden_size, out_features=(config.hidden_size + config.pos_size) * 2 + config.hidden_size, bias=False)
            # self.v_pos = nn.Linear(in_features=config.pos_size, out_features=config.pos_size, bias=False)
            # self.v_sem = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.wo_pos = nn.Linear(in_features=config.pos_size, out_features=config.pos_size, bias=False)
            self.wo_sem = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.resid_dropout = nn.Dropout(config.dropout_prob)
            
            # self.mix_fct = torch.mul if self.config.mix_attentions == "hadamard" else torch.add

            self.sem_attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.pos_attention_head_size = int(config.pos_size / config.num_attention_heads)

            if self.config.untie_cls :
                self.cls_sep_override = CLSSEPAttentionReplacer(self.config.num_attention_heads)
            # self.theta_sep_out = nn.Parameter(torch.tensor(0.5))
            # self.theta_sep_in  = nn.Parameter(torch.tensor(0.5))


        if self.config.relative_pos_bias :
            self.relative_pos_bias = RelativePositionBucketedBias(num_heads=self.config.num_attention_heads, 
                                                                max_seq_len=self.config.max_length,
                                                                num_buckets=32,
                                                                max_distance=128)
            
        self.attn_activation_fct = torch.softmax if self.config.attention_activation == "softmax" else softpick


        # Feedforward network
        match config.hidden_act.lower():
            case "swiglu":
                # To keep the number of parameters and the amount of computation constant, we reduce the number of
                # hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
                # avoid RuntimeError due to misaligned operand
                multiple_of = 8
                if not self.config.posneobert :
                    intermediate_size = int(2 * (config.intermediate_size) / 3)
                    intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
                    self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)
                else :
                    # FOR POSBERT
                    if self.config.mixed_feed_forward :
                        intermediate_size = int(2 * (config.pos_intermediate_size + config.intermediate_size) / 3)
                        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
                        # print("intermediate_size".upper())
                        # print(intermediate_size)
                        self.ffn = SwiGLU(config.hidden_size + config.pos_size, intermediate_size, config.hidden_size + config.pos_size, bias=False)
                    else :
                        pos_intermediate_size = int(2 * (config.pos_intermediate_size) / 3)
                        pos_intermediate_size = multiple_of * ((pos_intermediate_size + multiple_of - 1) // multiple_of)
                        self.pos_ffn = SwiGLU(config.pos_size, pos_intermediate_size, config.pos_size, bias=False)

                        sem_intermediate_size = int(2 * (config.intermediate_size) / 3)
                        sem_intermediate_size = multiple_of * ((sem_intermediate_size + multiple_of - 1) // multiple_of)
                        self.sem_ffn = SwiGLU(config.hidden_size, sem_intermediate_size, config.hidden_size, bias=False)
            
            case "posbertswiglu":
                multiple_of = 8
                # if self.config.mixed_feed_forward :
                intermediate_size = int(2 * (config.pos_intermediate_size + config.intermediate_size) / 3)
                intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
                self.ffn = PosBERTSwiGLU(config.hidden_size + config.pos_size, intermediate_size, config.hidden_size, config.pos_size, separate_w3=True )
                
            case "gelu":
                if not self.config.posneobert :
                    self.ffn = nn.Sequential(
                        nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                        nn.GELU(),
                        nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                    )
                else :
                    # FOR POSBERT
                    if self.config.mixed_feed_forward :
                        self.ffn = nn.Sequential(
                            nn.Linear(config.hidden_size + config.pos_size, config.intermediate_size + config.pos_intermediate_size, bias=False),
                            nn.GELU(),
                            nn.Linear(config.intermediate_size + config.pos_intermediate_size, config.hidden_size + config.pos_size, bias=False),
                        )
                    
                    else :
                        self.pos_ffn = nn.Sequential(
                            nn.Linear(config.pos_size,config.pos_intermediate_size, bias=False),
                            nn.GELU(),
                            nn.Linear(config.pos_intermediate_size, config.pos_size, bias=False),
                        )

                        self.sem_ffn = nn.Sequential(
                            nn.Linear(config.hidden_size,config.intermediate_size, bias=False),
                            nn.GELU(),
                            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
                        )


        # Pre-Layer Norm
        if not self.config.posneobert :
            self.attention_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
            self.ffn_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
        else :
            # separate LayerNorm
            self.sem_attention_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
            self.sem_ffn_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
            self.pos_attention_norm = (
                RMSNorm(config.pos_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.pos_size, config.norm_eps)
            )
            self.pos_ffn_norm = (
                RMSNorm(config.pos_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.pos_size, config.norm_eps)
            )

        
        # FFN dropout
        self.ffn_dropout = nn.Dropout(config.dropout_prob)


    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, shared_pos_keys: torch.Tensor | None = None):
        attn_weight = None
        pos_sem_weights = None
        if self.config.posneobert :
            #separated normalization
            # print("here is x", x)
            # print("cut as", x[..., :self.config.pos_size])

            x_pos = x[..., :self.config.pos_size]
            x_pos = self.pos_attention_norm(x_pos)
            x_sem = self.sem_attention_norm(x[..., self.config.pos_size:])
            x = torch.cat([x_pos, x_sem], dim=-1)
            # print("after 1", x)
            new_x, attn_weight, pos_sem_weights = self._posneobert_att_block(x=x, pad_mask=pad_mask, freqs_cis=freqs_cis, shared_pos_keys = shared_pos_keys)
            x = x + new_x
            # print("after attn block", x)
            x_pos = self.pos_ffn_norm(x[..., :self.config.pos_size])
            x_sem = self.sem_ffn_norm(x[..., self.config.pos_size:])
            x = torch.cat([x_pos, x_sem], dim=-1).contiguous()

            # print("input of ffblock", x.shape)
            x = x + self._posneobert_ff_block(x)
            # print("after ff", x)
            # print("x in forward", x)

        else :
            new_x, attn_weight = self._att_block(self.attention_norm(x), pad_mask, freqs_cis)
            x = x + new_x
            x = x + self._ff_block(self.ffn_norm(x))

        return x, attn_weight, pos_sem_weights

    def _att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        if self.config.attention_activation == "softpick" :
            raise NotImplementedError
        
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.qkv(x).view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3).chunk(3, axis=-1)

        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        pos_bias = self.relative_pos_bias(seq_len) if self.config.relative_pos_bias else None

        attn_weight = None
        # print("xqxk", xq, xk)
        if self.config.flash_attention:
            if pos_bias is not None :
                raise NotImplementedError
            # attn = memory_efficient_attention(query=xq, key=xk, value=xv, attn_bias=pad_mask, p=0)
        else:
            # Input and output are of dimension (B, H, M, K) (b_size, num_head, seqlength, h_dim)

            attn, attn_weight = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.dropout_prob if self.training else 0,
                attn_activation_fct = self.attn_activation_fct,
                RP_bias=pos_bias
            )
            attn = attn.transpose(1, 2)

        # print("attention", attn)
        return self.resid_dropout(self.wo(attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.config.dim_head))), attn_weight

    def _posneobert_att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor, shared_pos_keys: torch.Tensor | None = None):
        batch_size, seq_len, _ = x.shape

        # print("x shape", x.shape)
        if self.config.shared_pos_keys :
            raise NotImplementedError
            xq_pos = self.q_pos(x[..., :self.config.pos_size]).view(batch_size, seq_len, self.config.num_attention_heads, ((self.config.pos_size + self.config.hidden_size) // self.config.num_attention_heads))
            xk_pos = shared_pos_keys # is of shape [bs, sk, nh, hs] already, where all heads have the same data.
        # else :
        
        # 1. Project to the flat total dimension
        raw_qkv_pos = self.qkv_pos(x[..., :self.config.pos_size]) # [B, L, Total_Pos_Dim]
        raw_qkv_sem = self.qkv_sem(x[..., self.config.pos_size:]) # [B, L, Total_Sem_Dim]

        # print("after qkv", raw_qkv_pos)
        # 2. Split the total dimension (Dim -1)
        # Note: qk_dim here is the TOTAL dimension across all heads
        qk_dim = self.config.hidden_size + self.config.pos_size

        xq_pos, xk_pos, xv_pos = torch.split(raw_qkv_pos, [qk_dim, qk_dim, self.config.pos_size], dim=-1)
        xq_sem, xk_sem, xv_sem = torch.split(raw_qkv_sem, [qk_dim, qk_dim, self.config.hidden_size], dim=-1)

        # 3. Now reshape into heads
        # We calculate head_dim for each specifically
        head_qk = qk_dim // self.config.num_attention_heads
        head_v_pos = self.config.pos_size // self.config.num_attention_heads
        head_v_sem = self.config.hidden_size // self.config.num_attention_heads

        def reshape_heads(t, h_dim):
            return t.view(batch_size, seq_len, self.config.num_attention_heads, h_dim)

        xq_pos = reshape_heads(xq_pos, head_qk)
        xk_pos = reshape_heads(xk_pos, head_qk)
        xv_pos = reshape_heads(xv_pos, head_v_pos)

        xq_sem = reshape_heads(xq_sem, head_qk)
        xk_sem = reshape_heads(xk_sem, head_qk)
        xv_sem = reshape_heads(xv_sem, head_v_sem)

        if self.config.rope :
            xq_pos, xk_pos = apply_rotary_emb(xq_pos, xk_pos, freqs_cis)

        pos_bias = self.relative_pos_bias(seq_len) if self.config.relative_pos_bias else None # (batch_size, num_heads, seq_len, seq_len) or None

        # print("xqp, xkp, xqs, xks, xvp, xvs", xq_pos.shape, xk_pos.shape, xq_sem.shape, xk_sem.shape, xv_pos.shape, xv_sem.shape)

        if self.config.flash_attention:
            raise NotImplementedError
            #doesnt work as is
            # pos_attn = memory_efficient_attention(query=xq, key=xk, value=xv_pos, attn_bias=pad_mask, p=0) # (b_size, num_head, seqlength, pos_head_dim)
            # sem_attn = memory_efficient_attention(query=xq, key=xk, value=xv_sem, attn_bias=pad_mask, p=0) # (b_size, num_head, seqlength, sem_head_dim)
        else:
            # #TODO => make sure, but it seems that the dropout is the same for pos and sem
            
            # Input are of dimension (B, H, M, K) (b_size, num_head, seqlength, h_dim)
            # output are of dimension (B, H, M, M) (b_size, num_head, seqlength, seqlength)
            pos_attn_weight = posbert_scaled_dot_product_attention(
                query=xq_pos.transpose(1, 2),
                key=xk_pos.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.pos_dropout_prob if self.training else 0,
                attention_activation = self.config.attention_activation
            )


            sem_attn_weight = posbert_scaled_dot_product_attention(
                query=xq_sem.transpose(1, 2),
                key=xk_sem.transpose(1, 2),
                attn_mask=pad_mask,
                # dropout_p=self.config.pos_dropout_prob if self.training else 0,
                # attention_activation = self.config.attention_activation
            )

        # print("after weight", pos_attn_weight)
        if self.config.untie_cls :
            pos_attn_weight = self.cls_sep_override(pos_attn_weight, pad_mask)
        
        # print("after unit cls", pos_attn_weight)
        if self.config.mix_attentions == "sum" :
            attn_weight = torch.add(pos_attn_weight,sem_attn_weight)
            if self.config.relative_pos_bias :
                attn_weight = torch.add(attn_weight,pos_bias)
            attn_weight = self.attn_activation_fct(attn_weight,  dim=-1).to(xq_sem.dtype)

            # print("after softmax", attn_weight)
        elif self.config.mix_attentions == "hadamard" :
            raise NotImplementedError
            # pos_p = torch.softmax(pos_attn_weight, dim=-1)
            # sem_p = torch.softmax(sem_attn_weight, dim=-1)
            # if self.config.relative_pos_bias :
            #     attn_weight = self.attn_activation_fct(pos_p*sem_p + pos_bias, dim=-1).to(xq_sem.dtype)
            # else :
            #     attn_weight = self.attn_activation_fct(pos_p*sem_p, dim=-1).to(xq_sem.dtype)

            # print("after softpick", attn_weight)
            # print("attention weight", attn_weight.shape)

        attn_weight = torch.dropout(attn_weight, self.config.pos_dropout_prob if self.training else 0, train=True)

        xv_pos = xv_pos.reshape(batch_size, seq_len, self.config.num_attention_heads, self.pos_attention_head_size)
        xv_sem = xv_sem.reshape(batch_size, seq_len, self.config.num_attention_heads, self.sem_attention_head_size)

        pos_attn = (attn_weight @ xv_pos.transpose(1,2)).transpose(1, 2) # [b_size, seq_length, num_head, pos_size]
        sem_attn = (attn_weight @ xv_sem.transpose(1,2)).transpose(1, 2) # [b_size, seq_length, num_head, sem_size]

        pos_attn = self.wo_pos(pos_attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.pos_attention_head_size))
        sem_attn = self.wo_sem(sem_attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.sem_attention_head_size))
        attn = torch.cat([pos_attn, sem_attn], dim=-1).to(x.dtype)

        return self.resid_dropout(attn), attn_weight, [pos_attn_weight, sem_attn_weight, pos_bias]

    def _posneobert_ff_block(self, x:torch.Tensor):
        if self.config.hidden_act.lower() == "posbertswiglu" :
            return self.ffn_dropout(self.ffn(x))

        if self.config.mixed_feed_forward :
            x = self.ffn(x.clone().contiguous())
        else :
            x_pos = self.pos_ffn(x[..., :self.config.pos_size])
            x_sem = self.sem_ffn(x[..., self.config.pos_size:])
            x = torch.cat([x_pos, x_sem], dim=-1)

        return self.ffn_dropout(x)
    
    
    def _ff_block(self, x: torch.Tensor):
        return self.ffn_dropout(self.ffn(x))


    



class NormEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        self.attention_head_size = int((config.hidden_size + config.pos_size) / config.num_attention_heads)
        self.sem_attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.pos_attention_head_size = int(config.pos_size / config.num_attention_heads)

        self.all_head_size = config.num_attention_heads * self.attention_head_size
        self.sem_all_head_size = config.num_attention_heads * self.sem_attention_head_size
        self.pos_all_head_size = config.num_attention_heads * self.pos_attention_head_size

        # Attention
        if not self.config.posneobert :
            self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
            self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.resid_dropout = nn.Dropout(config.dropout)
        else :
            self.qk = nn.Linear(in_features=config.hidden_size + config.pos_size, out_features=(config.hidden_size + config.pos_size) * 2, bias=False)
            self.v_pos = nn.Linear(in_features=config.pos_size, out_features=config.pos_size, bias=False)
            self.v_pos = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
            self.pos_resid_dropout = nn.Dropout(config.pos_dropout_prob)
            self.sem_resid_dropout = nn.Dropout(config.dropout_prob)

        
        self.c_fc = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.ffn_dropout = nn.Dropout(config.dropout)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale
        self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling * torch.ones(config.hidden_size))

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(config.hidden_size))

        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(config.hidden_size))

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.suv = torch.nn.Parameter(self.suv_init_scaling * torch.ones(2 * config.intermediate_size))

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        x_attn = self._att_block(x, pad_mask, freqs_cis)

        lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_attn)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        x_ff = self._ff_block(x)

        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)

        A_norm = self.justnorm(x)
        B_norm = self.justnorm(x_ff)
        x = self.justnorm(A_norm + lr * (B_norm - A_norm))

        return x

    def _att_block(self, x: torch.Tensor, pad_mask: torch.Tensor, freqs_cis: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.qkv(x).view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3).chunk(3, axis=-1)

        if self.config.rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
            1, 1, self.config.num_attention_heads, self.config.hidden_size // self.config.num_attention_heads
        )
        xq = sqk * self.justnorm(xq)
        xk = sqk * self.justnorm(xk)

        softmax_scale = (self.config.hidden_size / self.config.num_attention_heads) ** 0.5

        if self.config.flash_attention:
            raise NotImplementedError
            # attn = memory_efficient_attention(query=xq, key=xk, value=xv, attn_bias=pad_mask, p=0, scale=softmax_scale)
        else:
            # Input and output are of dimension (B, H, M, K)
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=pad_mask,
                dropout_p=self.config.dropout_prob if self.training else 0,
                scale=softmax_scale,
            ).transpose(1, 2)

        return self.resid_dropout(self.wo(attn.reshape(batch_size, seq_len, self.config.hidden_size)))

    def _ff_block(self, x: torch.Tensor):
        uv = self.c_fc(x)
        suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.hidden_size**0.5))
        uv = suv * uv

        u, v = torch.chunk(uv, 2, dim=-1)
        x = u * self.silu(v)
        x = self.mlp_c_proj(x)

        return self.ffn_dropout(x)


class NeoBERTPreTrainedModel(PreTrainedModel):
    config_class = NeoBERTConfig
    _supports_cache_class = True

    def _init_weights(self, module):
        if getattr(module, "_skip_weight_init", False):
            return  #  Skip this one

        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)
        elif isinstance(module, CLSSEPAttentionReplacer):
            # Initialize head-specific thetas randomly
            module.thetas.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size + config.pos_size, config.hidden_size)
        # self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        # pooled_output = self.dense(first_token_tensor)
        # pooled_output = self.activation(pooled_output)
        return hidden_states[:, 0]



class NeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig, add_pooling_layer=True):
        super().__init__(config)

        self.config = config

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.shared_pos_encoder = nn.Linear(in_features=config.pos_size, out_features=(config.hidden_size + config.pos_size) // config.num_attention_heads) if config.shared_pos_keys else None

        if self.config.rope and not self.config.posneobert:
            self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        elif self.config.posneobert:
            match config.positional_embed_init :
                case "fixed" :
                    embs = torch.full((config.max_length + 1, config.pos_size), 1/math.sqrt(config.max_length))
                    # Register as a buffer: it moves with the model to GPU, but isn't a 'parameter'
                    self.register_buffer("pos_embs", embs)
                    # embs = torch.full((config.max_length + 1, config.pos_size), 1/math.sqrt(config.max_length))
                    # self.positional_embedding = nn.Embedding.from_pretrained(embs, freeze=True)
                    # self.positional_embedding._skip_weight_init = True
                case "random" :
                    self.positional_embedding = nn.Embedding(config.max_length + 1, config.pos_size, padding_idx=config.pad_token_id)
                case "2dim_cosine" :
                    pass
                    # embs = torch.zeros((config.max_length + 1, config.pos_size))
                    # rows = torch.arange(config.max_length + 1, dtype=torch.float32)
                    # angles = math.pi * rows / config.max_length
                    # embs[:, :2] = torch.stack([torch.cos(angles)/10, torch.sin(angles)/10], dim=1)
                    # self.positional_embedding = nn.Embedding.from_pretrained(embs, freeze=False)
                    # self.positional_embedding._skip_weight_init = True
                case "2dim":
                    pass
            if self.config.rope :
                self.freqs_cis = precompute_freqs_cis((self.config.pos_size + self.config.hidden_size) // config.num_attention_heads, config.max_length)

        
        elif self.config.AP_embeddings:
            self.positional_embedding = nn.Embedding(config.max_length + 1, config.hidden_size, padding_idx=config.pad_token_id)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        if not self.config.posneobert :
            self.layer_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
        else :
            self.sem_layer_norm = (
                RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
            )
            self.pos_layer_norm = (
                RMSNorm(config.pos_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.pos_size, config.norm_eps)
            )
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
            self,
        input_ids: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs
        ):
    #     input_ids: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     token_type_ids: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.Tensor] = None,
    #     head_mask: Optional[torch.Tensor] = None,
    #     inputs_embeds: Optional[torch.Tensor] = None,
    #     encoder_hidden_states: Optional[torch.Tensor] = None,
    #     encoder_attention_mask: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) :
        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        all_attentions = []
        all_hidden_states = []
        all_pos_sem_attentions = []

        if not isinstance(input_ids, torch.Tensor):
            features_dict = input_ids
            input_ids = features_dict.get("input_ids")
            attention_mask = features_dict.get("attention_mask")
            # Extract any other keys your NeoBERT uses (position_ids, etc.)
            position_ids = features_dict.get("position_ids")

        return_dict = kwargs.get("return_dict", None) if kwargs.get("return_dict", None) is not None else self.config.use_return_dict

        if attention_mask is not None:
            if not (attention_mask.dtype != torch.bool and 1.0 not in attention_mask)    :
                attention_mask = torch.where(attention_mask == 1, 0.0, float("-inf"))

            assert attention_mask.dtype != torch.bool and 1.0 not in attention_mask, "NeoBERT expects an additive attention_mask"
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.config.num_attention_heads, attention_mask.size(-1), 1)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(input_ids.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: input_ids.shape[1]]

        # Embedding
        x = self.encoder(input_ids)

        # Positional embedding
        
        if (not self.config.rope) and (not self.config.posneobert) and (self.config.AP_embeddings) :
            mask = input_ids.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            x += self.positional_embedding(incremental_indices)
    

        if self.config.posneobert :
            mask = input_ids.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            if self.config.positional_embed_init == "fixed" :
                positional_embed = F.embedding(incremental_indices, self.pos_embs)
            else :
                if self.training and self.config.random_offset:
                    valid_lengths = mask.sum(dim=1)  # How many non-pad tokens per example
                    max_offsets = (self.config.max_length - valid_lengths).clamp(min=0)
            
                    # Generate random offsets for all examples in a single call
                    random_offsets =  torch.randint(0, max_offsets.max() + 1, (len(max_offsets),)).to(mask.device)
                    random_offsets = random_offsets * (random_offsets <= max_offsets)

                    # Add the random offsets to the positional indices
                    incremental_indices += random_offsets.unsqueeze(1) * mask  # Apply offset only to non-pad tokens
                positional_embed = self.positional_embedding(incremental_indices)
            
            # if position_pca is not None :
            #     print("using pca")
            #     positional_embed = position_pca[incremental_indices].to(x.dtype)
                    
            x = torch.concat([positional_embed, x], dim=-1)


    
        # Transformer encoder

        shared_pos_keys = self.shared_pos_encoder(positional_embed).unsqueeze(2).expand(-1, -1, self.config.num_attention_heads, -1) if self.config.shared_pos_keys else None
        for layer in self.transformer_encoder:
            # print("getting in x", x)
            
            x, attention, pos_sem_attentions = layer(x, attention_mask, freqs_cis, shared_pos_keys = shared_pos_keys)

            all_hidden_states.append(x)
            all_attentions.append(attention)
            all_pos_sem_attentions.append(pos_sem_attentions)

        # Final normalization layer
        if not self.config.posneobert :
            x = self.layer_norm(x)
        else :
            x_pos = self.pos_layer_norm(x[..., :self.config.pos_size])
            x_sem = self.sem_layer_norm(x[..., self.config.pos_size:])
            x = torch.cat([x_pos, x_sem], dim=-1)

        pooled_output = self.pooler(x) if self.pooler is not None else None


        # print("return_dict?", return_dict)
        if not return_dict:
            return (x, pooled_output) + (all_hidden_states,)



        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=x,
            pooler_output=pooled_output,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=None,
        )
        # Return the output of the last hidden layer
        # return x, all_attentions, all_hidden_states, all_pos_sem_attentions


class NormNeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if self.config.rope:
            self.freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        else:
            self.positional_embedding = nn.Embedding(config.max_length + 1, config.hidden_size, padding_idx=config.pad_token_id)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(NormEncoderBlock(config))

        self.layer_norm = (
            RMSNorm(config.hidden_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.hidden_size, config.norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.num_hidden_layers))

        self.sz_init_value = 1.00
        self.sz_init_scaling = config.base_scale
        self.sz = torch.nn.Parameter(self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32))

    def forward(self, src, pad_mask=None):
        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if pad_mask is not None:
            assert pad_mask.dtype != torch.bool and 1.0 not in pad_mask, "NeoBERT expects an additive pad_mask"
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.config.num_attention_heads, pad_mask.size(-1), 1)

        # RoPE
        freqs_cis = None
        if self.config.rope:
            self.freqs_cis = self.freqs_cis.to(src.device, non_blocking=True)
            freqs_cis = self.freqs_cis[: src.shape[1]]

        # Embedding
        x = self.encoder(src)

        # Positional embedding
        if not self.config.rope:
            mask = src.ne(self.config.pad_token_id).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask  #
            incremental_indices = incremental_indices.long() + self.config.pad_token_id
            x += self.positional_embedding(incremental_indices)

        # Transformer encoder
        for layer in self.transformer_encoder:
            x = layer(x, pad_mask, freqs_cis)

        # Return the output of the last hidden layer
        return x


class NeoBERTLMHead(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.model = NormNeoBERT(config) if self.config.ngpt else NeoBERT(config)

        if not self.config.posneobert :
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        else :
            if self.config.use_only_sem_for_decoding :
                self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
            else :
                self.decoder = nn.Linear(config.hidden_size + config.pos_size, config.vocab_size)

        self.post_init()

    def forward(self, src, pad_mask=None, position_pca=None):

        hidden_representation, all_attentions, all_hidden_states, all_pos_sem_attentions = self.model.forward(src, pad_mask, position_pca)

        if not self.config.posneobert :
            logits = self.decoder(hidden_representation)
        else :
            if self.config.use_only_sem_for_decoding :
                logits = self.decoder(hidden_representation[..., self.config.pos_size:])
            else :
                logits = self.decoder(hidden_representation)

        return {"hidden_representation": hidden_representation, 
                "logits": logits, 
                "all_attentions": all_attentions,
                "all_hidden_states": all_hidden_states, 
                "all_pos_sem_attentions":all_pos_sem_attentions}


class PosOnlyNeoBERTLMHead(NeoBERTLMHead) :
     
    def load_state_dict(self, state_dict, strict=True, assign=False, layers=[], *model_args, **kwargs):

        out = super().load_state_dict(state_dict, strict=strict, assign=assign)

        # Now modify the weights
        with torch.no_grad():
            for layer in layers:
                self.model.transformer_encoder[layer].qk.weight[:, self.config.pos_size:] = 0

        return out
    

class SemOnlyNeoBERTLMHead(NeoBERTLMHead) :

    def load_state_dict(self, state_dict, strict=True, assign=False, layers=[], *model_args, **kwargs):

        out = super().load_state_dict(state_dict, strict=strict, assign=assign)

        # Now modify the weights
        with torch.no_grad():
            for layer in layers:
                self.model.transformer_encoder[layer].qk.weight[:, :self.config.pos_size] = 0

        return out

class NeoBERTForSequenceClassification(NeoBERTPreTrainedModel):

    def __init__(
        self,
        config: NeoBERTConfig,
        num_labels: int = None,
        classifier_dropout: float = 0.1,
        classifier_init_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(config)

        self.config = config

        self.num_labels = num_labels or config.num_labels
        self.classifier_dropout = classifier_dropout
        self.classifier_init_range = classifier_init_range

        self.neobert = NeoBERT(config)
        
        if not self.config.posneobert :
            s_size = self.config.hidden_size
        # else :
        #     s_size = self.config.hidden_size + self.config.pos_size

        else :
            s_size = self.config.hidden_size if self.config.use_only_sem_for_decoding else self.config.hidden_size + self.config.pos_size
        
        # s_size = self.config.hidden_size
        self.dense = nn.Linear(s_size, s_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(s_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        output = self.neobert.forward(input_ids, attention_mask)
        
        # if self.config.posneobert :
        #     x = all_hidden_states[-2][:, 0, :]
        # else :
        hidden_representation = output["last_hidden_state"]
        # print(hidden_representation)
        x = hidden_representation[:, 0, :]

        if self.config.use_only_sem_for_decoding :
                x = x[..., self.config.pos_size:]
        
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "ression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # print("uh return dict ?", return_dict)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_representation,
            attentions=None,
        )


class NeoBERTHFForSequenceClassification(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)

        self.model = NeoBERT(config)

        

        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.classifier_init_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        outputs = self.model.forward(input_ids, attention_mask)

        hidden_representation = outputs["last_hidden_state"]        
        x = hidden_representation[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "ression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_representation,
            attentions=None,
        )

class NeoBERTForTokenClassification(NeoBERTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.neobert = NeoBERT(config, add_pooling_layer=False)
        
        self.classifier_dropout = getattr(config, "classifier_dropout", 0.1)
        self.classifier_init_range = getattr(config, "classifier_init_range", 0.02)
        input_size = config.pos_size + config.hidden_size if config.posneobert else config.hidden_size
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(input_size, config.num_labels)

        self.pos_layer_norm = (
                RMSNorm(config.pos_size, config.norm_eps) if config.rms_norm else nn.LayerNorm(config.pos_size, config.norm_eps)
            )

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.neobert.forward(input_ids, attention_mask)

        hidden_representation = outputs["last_hidden_state"] 
        second_to_last_hidden_states =outputs["hidden_states"][-2]

        sequence_output = hidden_representation
        # sequence_output = outputs[0]

        # sequence_output[..., :48] = self.pos_layer_norm(second_to_last_hidden_states[..., :48])
        # bs, seqlen, 
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            # output = (logits,)x + all_hidden_states[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_representation,
            attentions=None,
        )


# class PosBERTForTokenClassification(nn.Module):
#     def __init__(self, encoder, hidden_size, pos_size, use_only_sem, num_labels, dropout_rate=0.1, use_second_to_last=False):
#         """
#         Initializes the model wrapper.
        
#         Args:
#             encoder (nn.Module): The instantiated base encoder model.
#             hidden_size (int): The dimensionality of the encoder's output features 
#                                (e.g., 768 for base models, 1024 for large models).
#             num_labels (int): The total number of BIO tags (15 for your PERSUADE setup).
#             dropout_rate (float): Regularization parameter to prevent overfitting.
#         """
#         super().__init__()
#         self.encoder = encoder

#         assert not (use_only_sem and use_second_to_last)
        
#         # Regularization is not optional.
#         self.dropout = nn.Dropout(dropout_rate)
#         self.hidden_size = hidden_size
#         self.pos_size = pos_size
#         self.use_second_to_last=use_second_to_last
#         if use_second_to_last :
#             self.pos_norm = RMSNorm(pos_size, 1e-6)
#         self.use_only_sem = use_only_sem
#         linear_input_size = hidden_size if use_only_sem else hidden_size+pos_size
#         # The linear transformation mapping token embeddings to class logits
#         self.classifier = nn.Linear(linear_input_size, num_labels)

#     def forward(self, input_ids, attention_mask=None, **kwargs):
#         """
#         The forward pass processing tokens through the encoder and the classification head.
#         """
#         # Execute the encoder. 
#         attention_mask_float = torch.where(attention_mask == 1, 0.0, float("-inf"))
#         # Passing kwargs allows flexibility if your encoder requires token_type_ids or other inputs.
#         sequence_output, _, all_hidden_states, _ = self.encoder(input_ids, attention_mask_float)
        
#         # Handle the output extraction robustly.
#         # # If your encoder returns a tuple/list of the 4 elements you listed:
#         # if isinstance(encoder_outputs, (tuple, list)):
#         #     # We only extract the first element: last_hidden_state
#         #     # Shape: (batch_size, sequence_length, hidden_size)
#         #     sequence_output = encoder_outputs[0]
            
#         # # If your encoder returns a HuggingFace-style ModelOutput dictionary/object:
#         # elif hasattr(encoder_outputs, 'last_hidden_state'):
#         #     sequence_output = encoder_outputs.last_hidden_state
            
#         # else:
#         #     raise ValueError("Encoder output format not recognized. Expected tuple or object with 'last_hidden_state'.")
            
#         # Apply dropout to the hidden states
#         if self.use_second_to_last :
#             second_to_last_pos_output = all_hidden_states[-2][..., :self.pos_size]
#             second_to_last_pos_output = self.pos_norm(second_to_last_pos_output)
#             sequence_output = torch.cat([second_to_last_pos_output, sequence_output[..., self.pos_size:]], dim=-1)
#         if self.use_only_sem :
#             sequence_output = sequence_output[..., self.pos_size:]    

                

#         sequence_output = self.dropout(sequence_output)
        
#         # Project hidden states to the label dimension
#         # Logits Shape: (batch_size, sequence_length, num_labels)
#         logits = self.classifier(sequence_output)
        
#         return logits