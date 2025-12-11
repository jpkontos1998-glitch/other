import torch
import time
import einops
from torch.nn.attention.flex_attention import flex_attention

qkv_proj = torch.nn.Linear(512, 3 * 512, device="cuda", dtype=torch.bfloat16)
x = torch.randn(1000, 200, 512, device="cuda", dtype=torch.bfloat16)

start_time = time.time()
for _ in range(100):
    qkv = qkv_proj(x)
    torch.cuda.synchronize()
end_time = time.time()
print("Projection elapsed time: {:.6f} seconds".format(end_time - start_time))

start_time = time.time()
for _ in range(100):
    q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=8).unbind(1)
    torch.cuda.synchronize()
end_time = time.time()
print("Rearrangement elapsed time: {:.6f} seconds".format(end_time - start_time))

# Measure time for memory-efficient attention
start_time = time.time()
with torch.backends.cuda.sdp_kernel(
    enable_flash=False, enable_math=False, enable_mem_efficient=True
):
    for _ in range(100):
        attn_v_mem_efficient = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
end_time = time.time()
print("Memory-efficient attention elapsed time: {:.6f} seconds".format(end_time - start_time))

# Measure time for flash attention
start_time = time.time()
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    for _ in range(100):
        attn_v_flash = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
end_time = time.time()
print("Flash attention elapsed time: {:.6f} seconds".format(end_time - start_time))

print("====================================")

start_time = time.time()
for _ in range(100):
    qkv = qkv_proj(x)
    torch.cuda.synchronize()
end_time = time.time()
print("Projection elapsed time: {:.6f} seconds".format(end_time - start_time))

start_time = time.time()
for _ in range(100):
    q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=8).unbind(1)
    torch.cuda.synchronize()
end_time = time.time()
print("Rearrangement elapsed time: {:.6f} seconds".format(end_time - start_time))

# Measure time for memory-efficient attention
start_time = time.time()
with torch.backends.cuda.sdp_kernel(
    enable_flash=False, enable_math=False, enable_mem_efficient=True
):
    for _ in range(100):
        attn_v_mem_efficient = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
end_time = time.time()
print("Memory-efficient attention elapsed time: {:.6f} seconds".format(end_time - start_time))

# Measure time for flash attention
start_time = time.time()
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    for _ in range(100):
        attn_v_flash = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
end_time = time.time()
print("Flash attention elapsed time: {:.6f} seconds".format(end_time - start_time))


def relative_positional(score, b, h, q_idx, kv_idx):
    return score + (q_idx - kv_idx)


fa = torch.compile(flex_attention)
for _ in range(10):
    attn_v_flex = fa(q, k, v, score_mod=relative_positional)
    torch.cuda.synchronize()

start_time = time.time()
for _ in range(100):
    attn_v_flex = fa(q, k, v, score_mod=relative_positional)
    torch.cuda.synchronize()
end_time = time.time()
print("Flex attention elapsed time: {:.6f} seconds".format(end_time - start_time))
