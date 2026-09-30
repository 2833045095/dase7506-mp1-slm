"""Student algorithm: a modernised, wider causal transformer.

Diagnosed limitation of the classroom baseline (width 128, depth 4, 1.09M
parameters, 2.10 test BPB): the whole budget is spent on a very small stack,
positions are supplied by an absolute learned embedding that only exists for
offsets 0..255 and cannot express relative distance, the residual stream is
normalised with LayerNorm, and the feed-forward block is a plain GELU MLP
with no gating.

Change implemented here, keeping the two interfaces of model.py unchanged:

  * rotary position embeddings (RoPE) applied to q and k, so attention logits
    depend on relative offset rather than on an absolute index;
  * pre-norm RMSNorm in place of LayerNorm;
  * a gated SwiGLU feed-forward block (gate/up/down, bias free);
  * bias-free linear projections and scaled residual output projections;
  * residual dropout, because the supplied corpus is small enough that the
    wider stack memorises it: at a 6,000-step schedule validation BPB bottoms
    out near step 3,000 and then rises while the training loss keeps falling;
  * a wider and deeper residual stack, sized to stay inside the 64 MiB
    uncompressed inference-asset budget.

Two config keys exist for the controlled experiments reported in REPORT.md and
have no effect on the submitted model: ``rope`` toggles RoPE off (the same
network is rebuilt with a learned absolute position embedding) and ``dropout``
sets the residual dropout rate (0 by default).

The model is strictly causal and keeps no state between calls: a prediction at
position t depends on ids[:, :t+1] only, and every call to predict_log_probs
starts from scratch.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Pre-norm RMS normalisation: no mean subtraction, no bias."""

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normalised = x.float() * torch.rsqrt(variance + self.eps)
        return (normalised * self.weight.float()).to(x.dtype)


def rope_tables(head_dim, context, base=10000.):
    """cos/sin tables of shape [context, head_dim // 2]."""
    inv_freq = 1. / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(context, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """Rotate even/odd feature pairs of q or k. x is [batch, heads, time, head_dim]."""
    time = x.shape[-2]
    cos, sin = cos[:time].to(x.dtype)[None, None], sin[:time].to(x.dtype)[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class Block(nn.Module):
    def __init__(self, width, heads, hidden, dropout=0.):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        hidden = self.norm2(x)
        return x + self.drop(self.down(F.silu(self.gate(hidden)) * self.up(hidden)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads, depth = config['width'], config['heads'], config['depth']
        hidden = config.get('hidden') or int(8 * width / 3)
        self.use_rope = config.get('rope', True)
        self.dropout = float(config.get('dropout', 0.))
        self.token = nn.Embedding(config['vocab'], width)
        self.embed_drop = nn.Dropout(self.dropout)
        self.blocks = nn.ModuleList([Block(width, heads, hidden, self.dropout) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        if self.use_rope:
            cos, sin = rope_tables(width // heads, self.context)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
        else:
            self.pos = nn.Embedding(self.context, width)
        self.apply(self.initialize)
        # Residual output projections start small so a deeper stack keeps the
        # residual stream at unit scale.
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=.02 / math.sqrt(2 * depth))
            nn.init.normal_(block.down.weight, std=.02 / math.sqrt(2 * depth))
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def features(self, ids):
        x = self.token(ids)
        cos = sin = None
        if self.use_rope:
            cos, sin = self.rope_cos, self.rope_sin
        else:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = self.embed_drop(x)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets.

        A prediction at position t uses ids[:, :t+1] and nothing later; all
        temporary state is local to this call, so independent windows, examples
        and scoring passes never share state.
        """
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    """Model factory used by train.py and evaluate.py."""
    return GPT(config)