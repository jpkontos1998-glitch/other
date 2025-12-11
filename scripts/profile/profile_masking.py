import torch
import torch.nn as nn
import time
import einops
from torch.amp import autocast


class CachedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, k, v):
        """
        x: [batch, seq, d_model]
        """
        q = self.q_proj(x)
        q = einops.rearrange(q, "b t (k h d) -> b k h t d", k=1, h=self.n_head).unbind(1)[0]
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        except RuntimeError as e:
            if e.args[0] != "No available kernel.  Aborting execution.":
                raise e
            print("*** USING SLOW SDP KERNEL ***")
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, is_causal=False, attn_mask=None, **kwargs):
        """
        x: [batch, seq, d_model]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=self.n_head).unbind(1)
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                attn_v = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=is_causal
                )
        except RuntimeError as e:
            if e.args[0] != "No available kernel.  Aborting execution.":
                raise e
            print("*** USING SLOW SDP KERNEL ***")
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                attn_v = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout, ff_factor=4, use_cached_mha=False):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        if use_cached_mha:
            self.mha = CachedMultiHeadAttention(d_model, n_head)
        else:
            self.mha = MultiHeadAttention(d_model, n_head)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_factor * d_model)
        self.linear2 = nn.Linear(ff_factor * d_model, d_model)

        self.dropout = dropout

    def forward(self, x, **kwargs):
        x = x + nn.functional.dropout(self.mha(self.layer_norm1(x), **kwargs), p=self.dropout)
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x

    def _ff_block(self, x):
        x = self.linear2(nn.functional.relu(self.linear1(x)))
        return x


rollout_batch_size = 1024
n_token = 92
d_model = 384
seq_len = 2000
n_iter = 5
normal_layer = TransformerLayer(d_model, 8, 0.0).to("cuda")
x = torch.randn(rollout_batch_size, n_token, d_model, device="cuda", dtype=torch.bfloat16)

for _ in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
        normal_layer(x)
t = time.time()
torch.cuda.synchronize()
for _ in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
        normal_layer(x)
torch.cuda.synchronize()
print(f"Rollout spatial layer time: {(time.time() - t) / n_iter} for {x.shape}")
x = torch.randn(seq_len, n_token, d_model, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
for _ in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        normal_layer(x)
torch.cuda.synchronize()
print(f"Train spatial layer time: {(time.time() - t) / n_iter} for {x.shape}")

cached_layer = TransformerLayer(d_model, 8, 0.0, use_cached_mha=True).to("cuda")
x = torch.randn(rollout_batch_size, 1, d_model, device="cuda", dtype=torch.bfloat16)
k, v = (
    torch.randn(rollout_batch_size, 8, 100, d_model // 8, device="cuda", dtype=torch.bfloat16),
    torch.randn(rollout_batch_size, 8, 100, d_model // 8, device="cuda", dtype=torch.bfloat16),
)

for _ in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        cached_layer(x, k=k, v=v)
t = time.time()
torch.cuda.synchronize()
for _ in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
        cached_layer(x, k=k, v=v)
torch.cuda.synchronize()
print(f"Cached temporal layer time: {(time.time() - t) / n_iter} for {x.shape}")

mask = torch.randn(rollout_batch_size, seq_len, seq_len, device="cuda", dtype=torch.bfloat16)
x = torch.randn(1, seq_len, d_model, device="cuda", dtype=torch.bfloat16)
for _ in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        normal_layer(x, attn_mask=mask)
t = time.time()
torch.cuda.synchronize()
for _ in range(n_iter):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        normal_layer(x, attn_mask=mask)
torch.cuda.synchronize()
print(f"Training temporal layer time: {(time.time() - t) / n_iter} for {x.shape}")
