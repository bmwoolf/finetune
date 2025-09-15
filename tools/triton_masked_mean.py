import torch
import triton
import triton.language as tl

# Triton kernel: masked mean over L for each (b, d-tile)
# H: [B, L, D] float32
# M: [B, L]    int/bool (1=token, 0=pad)
# O: [B, D]    float32
@triton.jit
def _masked_mean_kernel(H_ptr, M_ptr, O_ptr,
                        B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                        stride_b_ld: tl.constexpr, stride_l_d: tl.constexpr, stride_d: tl.constexpr,
                        stride_b_l: tl.constexpr,
                        BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr):
    b = tl.program_id(0)          # batch index
    d_block = tl.program_id(1)    # feature-tile index

    d_idxs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_idxs < D

    # base pointers for this batch item
    H_b_ptr = H_ptr + b * stride_b_ld
    M_b_ptr = M_ptr + b * stride_b_l

    # accumulators across L
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    count = tl.zeros([1], dtype=tl.float32)

    # sweep L in tiles
    for l_start in range(0, L, BLOCK_L):
        l_idxs = l_start + tl.arange(0, BLOCK_L)
        l_ok = l_idxs < L

        # load mask chunk [BLOCK_L]
        m = tl.load(M_b_ptr + l_idxs, mask=l_ok, other=0).to(tl.float32)  # 1 or 0

        # broadcast to [BLOCK_L, BLOCK_D]
        m2 = m[:, None] * tl.where(d_mask[None, :], 1.0, 0.0)

        # load hidden chunk [BLOCK_L, BLOCK_D]
        H_ld_ptr = H_b_ptr + l_idxs[:, None] * stride_l_d + d_idxs[None, :] * stride_d
        h = tl.load(H_ld_ptr, mask=(l_ok[:, None] & d_mask[None, :]), other=0.0)

        # accumulate masked sums and token counts
        acc += tl.sum(h * m2, axis=0)
        count += tl.sum(m, axis=0)

    # avoid div by zero
    denom = tl.maximum(count, 1.0)
    out = acc / denom

    # store [BLOCK_D] to O[b, d_idxs]
    O_bd_ptr = O_ptr + b * D + d_idxs
    tl.store(O_bd_ptr, out, mask=d_mask)

# masked mean pool wrapper
def masked_mean_pool(hidden: torch.Tensor,
                     attention_mask: torch.Tensor,
                     *,
                     block_d: int = 128,
                     block_l: int = 256,
                     num_warps: int = 4,
                     num_stages: int = 2) -> torch.Tensor:
    """
    Triton-accelerated masked mean over sequence dimension.

    hidden: [B, L, D] float32 (contiguous or will be viewed as such)
    attention_mask: [B, L] int/bool (1=token, 0=pad)
    returns: [B, D] float32
    """
    assert hidden.ndim == 3, "hidden must be [B, L, D]"
    assert attention_mask.ndim == 2, "attention_mask must be [B, L]"
    B, L, D = hidden.shape
    assert attention_mask.shape[0] == B and attention_mask.shape[1] == L

    # ensure contiguous layout we expect: [B, L, D]
    H = hidden.contiguous()
    M = attention_mask.contiguous()
    O = torch.empty((B, D), device=H.device, dtype=H.dtype)

    # strides in elements (PyTorch strides are in elements)
    stride_b_ld = H.stride(0)
    stride_l_d  = H.stride(1)
    stride_d    = H.stride(2)
    stride_b_l  = M.stride(0)

    grid = (B, triton.cdiv(D, block_d))
    _masked_mean_kernel[grid](
        H, M, O,
        B, L, D,
        stride_b_ld, stride_l_d, stride_d,
        stride_b_l,
        BLOCK_D=block_d, BLOCK_L=block_l,
        num_warps=num_warps, num_stages=num_stages,
    )
    return O
