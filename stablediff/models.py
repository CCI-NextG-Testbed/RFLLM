import math
import re
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

import complex.complex_module as cm


def init_weight_norm(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_step, embed_dim), persistent=False)
        self.projection = nn.Sequential(
            cm.ComplexLinear(embed_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim, bias=True),
        )
        self.hidden_dim = hidden_dim
        self.apply(init_weight_norm)

    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64]:
            x = self.embedding[t]
        else:
            x = self._lerp_embedding(t)
        return self.projection(x)

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)  # [T, 1]
        dims = torch.arange(embed_dim).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step)
                                  * dims / embed_dim)  # [T, E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_len, hidden_dim), persistent=False)
        self.projection = cm.ComplexLinear(input_dim, hidden_dim)
        self.apply(init_weight_xavier)

    def forward(self, x): 
        x = self.projection(x)
        return cm.complex_mul(x, self.embedding.to(x.device))

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len)
                                  * dims / hidden_dim)     # [P,E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = cm.CosineComplexMultiHeadAttention(
            hidden_dim, num_heads, bias=True, **block_kwargs)
        self.norm_cross = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.cross_attn = cm.ComplexMultiHeadAttention(
            query_size=hidden_dim,
            num_hiddens=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
        )
        self.norm2 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 9*hidden_dim, bias=True)
        )
        self.apply(init_weight_xavier)
        self.adaLN_modulation.apply(init_weight_zero)

    def forward(self, x, c, text_tokens=None):
        """
        Embedding diffusion step t with adaptive layer-norm.
        Embedding condition c with cross-attention.
        - Input:\\
          x, [B, N, H, 2], \\ 
          t, [B, H, 2], \\
          c, [B, N, H, 2], \\
        """
        shift_msa, scale_msa, gate_msa, shift_xattn, scale_xattn, gate_xattn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(9, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + \
            gate_msa.unsqueeze(
                1) * self.attn(mod_x, mod_x, mod_x)

        if text_tokens is not None:
            mod_x_cross = modulate(self.norm_cross(x), shift_xattn, scale_xattn)
            x = x + gate_xattn.unsqueeze(1) * self.cross_attn(mod_x_cross, text_tokens, text_tokens)

        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.apply(init_weight_zero)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x
    
class tfdiff_Simple(nn.Module):
    """
    A simpler variant of tfdiff_WiFi for generic complex-valued sequences.

    Expected tensor shapes:
      x: [B, N, input_dim, 2]        (complex stored as (..., 2) = (real, imag))
      t: [B] or [B,] int/float step  (same as your DiffusionEmbedding usage)
      cond (optional):
         - None: uses only diffusion embedding t
         - tensor [B, cond_dim] real: projected into complex [B, H, 2]
         - tensor [B, H, 2] complex: used directly as conditioning
    Output:
      y: [B, N, output_dim, 2]
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = torch.device("cpu")

        self.input_dim = params.input_dim
        self.output_dim = getattr(params, "output_dim", params.input_dim)
        self.hidden_dim = params.hidden_dim
        self.num_heads = params.num_heads
        self.dropout = params.dropout
        self.mlp_ratio = params.mlp_ratio
        self.max_text_tokens = int(getattr(params, "max_text_tokens", 12))
        self.mod_embed_dim = int(getattr(params, "mod_embed_dim", 16))
        self.mod_to_id = {
            "BPSK": 0,
            "QPSK": 1,
            "8PSK": 2,
            "16QAM": 3,
            "64QAM": 4,
            "256QAM": 5,
        }

        # Embeddings
        self.p_embed = PositionEmbedding(params.sample_rate, self.input_dim, self.hidden_dim)
        self.t_embed = DiffusionEmbedding(params.max_step, params.embed_dim, self.hidden_dim)

        # Optional conditioning projection (real -> complex hidden)
        self.text_encoder = SentenceTransformer("intfloat/e5-large-v2")
        text_dim = self.text_encoder.get_sentence_embedding_dimension()
        # project real text embedding(s) to complex [B, H, 2] / [B, L, H, 2]
        self.text_proj = nn.Linear(text_dim, self.hidden_dim * 2)
        self.text_token_proj = nn.Linear(text_dim, self.hidden_dim * 2)
        init_weight_xavier(self.text_proj)
        init_weight_xavier(self.text_token_proj)

        self.mod_embedding = nn.Embedding(len(self.mod_to_id), self.mod_embed_dim)
        self.mod_router = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, len(self.mod_to_id)),
        )
        self.bits_token = nn.Linear(1 + self.mod_embed_dim, self.hidden_dim * 2)
        init_weight_xavier(self.mod_router[0])
        init_weight_xavier(self.mod_router[2])
        init_weight_xavier(self.bits_token)

        # populated on forward for auxiliary loss/inspection
        self.last_mod_logits = None
        self.last_mod_probs = None

        # Blocks + head
        self.blocks = nn.ModuleList(
            [DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(params.num_block)]
        )
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)

    def _encode_text(self, prompts, device):
        """
        prompts: list[str] or already a tensor
        Returns: complex conditioning vector [B, H, 2]
        """
        if isinstance(prompts, (list, tuple)):
            # SentenceTransformer handles batching internally
            text_emb = self.text_encoder.encode(
                prompts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )   # [B, D_text], real
        elif isinstance(prompts, torch.Tensor):
            # assume already [B, D_text] real embeddings
            text_emb = prompts.to(device)
        else:
            # single string
            text_emb = self.text_encoder.encode(
                [prompts],
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
            )   # [1, D_text]

        B = text_emb.shape[0]
        # project to 2*hidden_dim and reshape to complex [B, H, 2]
        text_proj = self.text_proj(text_emb)              # [B, 2H]
        text_proj = text_proj.view(B, self.hidden_dim, 2) # [B, H, 2]
        return text_proj

    def _prompt_words(self, prompt):
        words = re.findall(r"[A-Za-z0-9_+\-]+", str(prompt).lower())
        if len(words) == 0:
            words = ["unknown"]
        return words[:self.max_text_tokens]

    def _encode_text_tokens(self, prompts, device):
        """
        Returns:
          text_tokens: [B, L, H, 2]
          text_mask:   [B, L] bool
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, tuple):
            prompts = list(prompts)

        if isinstance(prompts, torch.Tensor):
            # Fallback: a precomputed sentence embedding tensor [B, D_text].
            sent = prompts.to(device)
            if sent.ndim == 1:
                sent = sent.unsqueeze(0)
            B = sent.shape[0]
            tok = self.text_token_proj(sent).view(B, 1, self.hidden_dim, 2)
            mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            return tok, mask

        prompt_words = [self._prompt_words(p) for p in prompts]
        B = len(prompt_words)
        L = max(len(ws) for ws in prompt_words)

        flat_words = []
        for ws in prompt_words:
            flat_words.extend(ws)

        word_emb = self.text_encoder.encode(
            flat_words,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
        )  # [sum(L_i), D_text]

        D = word_emb.shape[-1]
        token_emb = torch.zeros(B, L, D, dtype=word_emb.dtype, device=device)
        token_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        cursor = 0
        for i, ws in enumerate(prompt_words):
            li = len(ws)
            token_emb[i, :li, :] = word_emb[cursor:cursor+li, :]
            token_mask[i, :li] = True
            cursor += li

        token_proj = self.text_token_proj(token_emb)  # [B,L,2H]
        token_proj = token_proj.view(B, L, self.hidden_dim, 2)  # [B,L,H,2]
        return token_proj, token_mask
    
    def _modulation_context_from_prompt(self, c_prompt):
        """
        c_prompt: [B,H,2] complex-like tensor
        returns:
          mod_context: [B,Dm]
          mod_logits: [B,num_mods]
          mod_probs: [B,num_mods]
        """
        B = c_prompt.shape[0]
        c_real = c_prompt.reshape(B, -1)  # [B,2H]
        mod_logits = self.mod_router(c_real)
        mod_probs = F.softmax(mod_logits, dim=-1)
        mod_context = torch.matmul(mod_probs, self.mod_embedding.weight)  # [B,Dm]
        return mod_context, mod_logits, mod_probs

    def _encode_bits_seq(self, bits, mod_context, device, N):
        """
        bits: [B,N] normalized symbol-conditioning sequence
        mod_context: [B,Dm] soft modulation embedding inferred from prompt
        return: [B,N,H,2]
        """
        if bits is None:
            return None
        if not isinstance(bits, torch.Tensor):
            bits = torch.tensor(bits, dtype=torch.float32, device=device)
        else:
            bits = bits.to(device).float()

        if bits.ndim == 1:
            bits = bits.unsqueeze(0)  # [1,N]
        if bits.shape[1] != N:
            # If mismatch, you need a mapping from samples->symbols (oversampling etc.)
            # For now, truncate/pad as a safe fallback:
            if bits.shape[1] > N:
                bits = bits[:, :N]
            else:
                pad = torch.zeros(bits.shape[0], N - bits.shape[1], device=device)
                bits = torch.cat([bits, pad], dim=1)

        bits = bits.unsqueeze(-1)  # [B,N,1]
        B = bits.shape[0]
        if mod_context is None:
            mod_context = torch.zeros(B, self.mod_embed_dim, device=device, dtype=bits.dtype)
        mod_seq = mod_context.unsqueeze(1).expand(B, N, self.mod_embed_dim)  # [B,N,Dm]
        bits_feat = torch.cat([bits, mod_seq], dim=-1)          # [B,N,1+Dm]
        b = self.bits_token(bits_feat)              # [B,N,2H]
        b = b.view(B, N, self.hidden_dim, 2)   # [B,N,H,2]
        return b

    def forward(self, x, t, cond):
        device = x.device

        # cond is dict: {'prompt': label/list[str], 'bits_cond': [B,N]}
        prompt_input = cond.get("prompt") if isinstance(cond, dict) else cond
        bits_input   = cond.get("bits_cond", cond.get("bits")) if isinstance(cond, dict) else None

        # x expected [B,N,1,2]
        B, N = x.shape[0], x.shape[1]

        # tokenize signal
        x = self.p_embed(x)        # [B,N,H,2]

        # timestep embedding
        t = self.t_embed(t)        # [B,H,2]

        # prompt now has both:
        # 1) global conditioning vector c
        # 2) token sequence for explicit cross-attention
        text_tokens, text_mask = self._encode_text_tokens(prompt_input, device)  # [B,L,H,2], [B,L]
        if text_tokens.shape[0] == 1 and B > 1:
            text_tokens = text_tokens.expand(B, -1, -1, -1)
            text_mask = text_mask.expand(B, -1)

        text_mask_f = text_mask.to(text_tokens.dtype).unsqueeze(-1).unsqueeze(-1)  # [B,L,1,1]
        denom = text_mask_f.sum(dim=1).clamp(min=1.0)
        c_prompt = (text_tokens * text_mask_f).sum(dim=1) / denom  # [B,H,2]
        c = t + c_prompt

        # Prompt-driven modulation routing (soft, learned).
        mod_context, mod_logits, mod_probs = self._modulation_context_from_prompt(c_prompt)
        self.last_mod_logits = mod_logits
        self.last_mod_probs = mod_probs

        # bits are per-token injection (unambiguous)
        b_seq = self._encode_bits_seq(bits_input, mod_context, device, N)  # [B,N,H,2] or None
        if b_seq is not None:
            x = x + b_seq

        for block in self.blocks:
            x = block(x, c, text_tokens=text_tokens)

        x = self.final_layer(x, c)
        return x
