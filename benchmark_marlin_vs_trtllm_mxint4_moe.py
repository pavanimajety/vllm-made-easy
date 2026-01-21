#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark comparing Marlin INT4 MoE vs FlashInfer TRT-LLM MXINT4 MoE.

Usage:
    python benchmarks/benchmark_marlin_vs_trtllm_mxint4_moe.py

Environment variables:
    VLLM_USE_FLASHINFER_MOE_INT4=1  # Required for TRT-LLM path
"""

import argparse
import os
import time
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    grouped_topk,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


def mxint4_quantize(
    x: torch.Tensor, sf_vec_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 tensor to MXINT4 with block scaling (group_size=sf_vec_size)."""
    x_reshaped = x.reshape(-1, sf_vec_size)
    x_max = x_reshaped.max(dim=-1, keepdim=True)[0].to(torch.float32)
    x_min = x_reshaped.min(dim=-1, keepdim=True)[0].to(torch.float32)
    x_max = x_max * 8.0 / 7.0
    amax = torch.where(x_max > -x_min, x_max, -x_min)
    scales = amax / 8.0
    x_scaled = x_reshaped * scales.reciprocal()
    x_int8 = (
        x_scaled.round().clamp(-8, 7).to(torch.int8).reshape(-1, sf_vec_size // 2, 2)
    )
    x_int4 = (x_int8[..., 0] & 0x0F) | ((x_int8[..., 1] & 0x0F) << 4)
    return (
        x_int4.to(torch.uint8).reshape(*x.shape[:-1], x.shape[-1] // 2),
        scales.to(x.dtype).reshape(*x.shape[:-1], x.shape[-1] // sf_vec_size),
    )


def mxint4_quantize_moe_weights(
    weights_bf16: torch.Tensor, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize MoE weights [e, n, k] to MxInt4 format."""
    e = weights_bf16.shape[0]
    weight_list = []
    scale_list = []

    for i in range(e):
        w_q, w_s = mxint4_quantize(weights_bf16[i], sf_vec_size=group_size)
        weight_list.append(w_q)
        scale_list.append(w_s)

    return torch.stack(weight_list), torch.stack(scale_list)


def marlin_quantize_moe_weights(
    weights_bf16: torch.Tensor, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize MoE weights [e, n, k] to Marlin INT4 format."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )

    e, n, k = weights_bf16.shape
    weight_list = []
    scale_list = []

    for i in range(e):
        w_t = weights_bf16[i].T.contiguous()
        _, w_q, w_s, _, _, _ = marlin_quantize(
            w_t, scalar_types.uint4b8, group_size, act_order=False
        )
        weight_list.append(w_q)
        scale_list.append(w_s)

    weights_marlin = torch.stack(weight_list)
    scales_marlin = torch.stack(scale_list)

    return weights_marlin, scales_marlin


def benchmark_marlin_moe(
    a: torch.Tensor,
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    w1_marlin: torch.Tensor,
    w2_marlin: torch.Tensor,
    w1_scales_marlin: torch.Tensor,
    w2_scales_marlin: torch.Tensor,
    e: int,
    topk: int,
    n_group: int,
    topk_group: int,
    routed_scaling: float,
    dtype: torch.dtype,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> dict[str, Any]:
    """Benchmark Marlin INT4 MoE kernel with production routing.
    
    Times the full path: grouped_topk (routing) + fused_marlin_moe (computation).
    """
    # Warmup
    for _ in range(num_warmup):
        topk_weights, topk_ids = grouped_topk(
            hidden_states=a,
            gating_output=routing_logits,
            topk=topk,
            renormalize=False,
            num_expert_group=n_group,
            topk_group=topk_group,
            scoring_func="sigmoid",
            routed_scaling_factor=routed_scaling,
            e_score_correction_bias=routing_bias,
        )
        _ = fused_marlin_moe(
            a,
            w1_marlin,
            w2_marlin,
            None,
            None,
            w1_scales_marlin,
            w2_scales_marlin,
            None,
            topk_weights,
            topk_ids,
            global_num_experts=e,
            expert_map=None,
            global_scale1=None,
            global_scale2=None,
            g_idx1=None,
            g_idx2=None,
            input_global_scale1=None,
            input_global_scale2=None,
            sort_indices1=None,
            sort_indices2=None,
            w1_zeros=None,
            w2_zeros=None,
            input_dtype=dtype,
            quant_type_id=scalar_types.uint4b8.id,
            is_k_full=True,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        # Production path: routing + computation
        topk_weights, topk_ids = grouped_topk(
            hidden_states=a,
            gating_output=routing_logits,
            topk=topk,
            renormalize=False,
            num_expert_group=n_group,
            topk_group=topk_group,
            scoring_func="sigmoid",
            routed_scaling_factor=routed_scaling,
            e_score_correction_bias=routing_bias,
        )
        output = fused_marlin_moe(
            a,
            w1_marlin,
            w2_marlin,
            None,
            None,
            w1_scales_marlin,
            w2_scales_marlin,
            None,
            topk_weights,
            topk_ids,
            global_num_experts=e,
            expert_map=None,
            global_scale1=None,
            global_scale2=None,
            g_idx1=None,
            g_idx2=None,
            input_global_scale1=None,
            input_global_scale2=None,
            sort_indices1=None,
            sort_indices2=None,
            w1_zeros=None,
            w2_zeros=None,
            input_dtype=dtype,
            quant_type_id=scalar_types.uint4b8.id,
            is_k_full=True,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    latency_ms = (end - start) * 1000 / num_iters
    return {
        "latency_ms": latency_ms,
        "throughput_toks_per_sec": 1000.0 / latency_ms * a.shape[0],
        "output": output,
    }


def benchmark_trtllm_moe(
    a: torch.Tensor,
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
    e: int,
    n: int,
    topk: int,
    n_group: int,
    topk_group: int,
    routed_scaling: float,
    dtype: torch.dtype,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> dict[str, Any]:
    """Benchmark TRT-LLM MXINT4 MoE kernel."""
    from flashinfer import RoutingMethodType
    from flashinfer.fused_moe import trtllm_mxint4_block_scale_moe

    # Warmup
    for _ in range(num_warmup):
        _ = trtllm_mxint4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias.to(torch.bfloat16),
            hidden_states=a,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_scales,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_scales,
            num_experts=e,
            top_k=topk,
            n_group=n_group,
            topk_group=topk_group,
            intermediate_size=n,
            local_expert_offset=0,
            local_num_experts=e,
            routed_scaling_factor=routed_scaling,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            enable_pdl=None,
            output=None,
            tune_max_num_tokens=8192,
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output = trtllm_mxint4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias.to(torch.bfloat16),
            hidden_states=a,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_scales,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_scales,
            num_experts=e,
            top_k=topk,
            n_group=n_group,
            topk_group=topk_group,
            intermediate_size=n,
            local_expert_offset=0,
            local_num_experts=e,
            routed_scaling_factor=routed_scaling,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            enable_pdl=None,
            output=None,
            tune_max_num_tokens=8192,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()

    latency_ms = (end - start) * 1000 / num_iters
    return {
        "latency_ms": latency_ms,
        "throughput_toks_per_sec": 1000.0 / latency_ms * a.shape[0],
        "output": output.to(dtype),
    }


def run_benchmark(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
    num_warmup: int,
    num_iters: int,
) -> dict[str, Any]:
    """Run benchmark for given configuration."""
    torch.cuda.manual_seed(0)
    dtype = torch.bfloat16

    # DeepSeekV3 routing config
    n_group = 1
    topk_group = 1
    routed_scaling = 2.827

    # Generate inputs
    a = torch.randn((m, k), device="cuda", dtype=dtype) * 0.5
    routing_logits = torch.randn((m, e), device="cuda", dtype=torch.float32) * 1.5
    routing_bias = torch.randn(e, device="cuda", dtype=torch.float32) * 0.8

    # Generate BF16 weights
    std_w1 = (2.0 / (k + 2 * n)) ** 0.5
    std_w2 = (2.0 / (n + k)) ** 0.5
    w1_bf16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) * std_w1
    w2_bf16 = torch.randn((e, k, n), device="cuda", dtype=dtype) * std_w2

    # === Marlin path: grouped_topk + fused_marlin_moe ===
    print("  Quantizing weights for Marlin...")
    w1_marlin, w1_scales_marlin = marlin_quantize_moe_weights(w1_bf16, group_size)
    w2_marlin, w2_scales_marlin = marlin_quantize_moe_weights(w2_bf16, group_size)

    print("  Benchmarking Marlin (routing + computation)...")
    marlin_results = benchmark_marlin_moe(
        a,
        routing_logits,
        routing_bias,
        w1_marlin,
        w2_marlin,
        w1_scales_marlin,
        w2_scales_marlin,
        e,
        topk,
        n_group,
        topk_group,
        routed_scaling,
        dtype,
        num_warmup,
        num_iters,
    )

    # === TRT-LLM path: trtllm_mxint4_block_scale_moe (fused routing + computation) ===
    print("  Quantizing and preparing weights for TRT-LLM...")
    w1_int4, w1_scales = mxint4_quantize_moe_weights(w1_bf16, group_size)
    w2_int4, w2_scales = mxint4_quantize_moe_weights(w2_bf16, group_size)
    
    from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (
        prepare_static_weights_for_trtllm_mxint4_moe,
    )
    
    trtllm_weights = prepare_static_weights_for_trtllm_mxint4_moe(
        gemm1_weights=w1_int4,
        gemm1_scales=w1_scales,
        gemm2_weights=w2_int4,
        gemm2_scales=w2_scales,
    )

    print("  Benchmarking TRT-LLM (fused routing + computation)...")
    trtllm_results = benchmark_trtllm_moe(
        a,
        routing_logits,
        routing_bias,
        trtllm_weights["gemm1_weights"],
        trtllm_weights["gemm1_scales"],
        trtllm_weights["gemm2_weights"],
        trtllm_weights["gemm2_scales"],
        e,
        n,
        topk,
        n_group,
        topk_group,
        routed_scaling,
        dtype,
        num_warmup,
        num_iters,
    )

    # Verify correctness
    max_diff = torch.abs(marlin_results["output"] - trtllm_results["output"]).max()
    mean_diff = torch.abs(marlin_results["output"] - trtllm_results["output"]).mean()

    return {
        "m": m,
        "n": n,
        "k": k,
        "e": e,
        "topk": topk,
        "group_size": group_size,
        "marlin_latency_ms": marlin_results["latency_ms"],
        "marlin_throughput": marlin_results["throughput_toks_per_sec"],
        "trtllm_latency_ms": trtllm_results["latency_ms"],
        "trtllm_throughput": trtllm_results["throughput_toks_per_sec"],
        "speedup": marlin_results["latency_ms"] / trtllm_results["latency_ms"],
        "max_diff": max_diff.item(),
        "mean_diff": mean_diff.item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Marlin vs TRT-LLM MXINT4 MoE"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64, 128, 256, 512, 1024],
        help="Number of tokens (m) to benchmark",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=7168,
        help="Intermediate size (n)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=512, help="Hidden size (k)"
    )
    parser.add_argument(
        "--num-experts", type=int, default=384, help="Number of experts (e)"
    )
    parser.add_argument("--topk", type=int, default=8, help="Top-K experts")
    parser.add_argument(
        "--group-size", type=int, default=32, help="Quantization group size"
    )
    parser.add_argument(
        "--num-warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-iters", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Output CSV file path"
    )

    args = parser.parse_args()

    if current_platform.is_rocm():
        print("Skipping benchmark on ROCm platform")
        return

    # Check for flashinfer
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        print("FlashInfer not installed. Install with: pip install flashinfer")
        return

    # Enable TRT-LLM path
    os.environ["VLLM_USE_FLASHINFER_MOE_INT4"] = "1"

    print("=" * 80)
    print("Benchmark: Marlin INT4 MoE vs TRT-LLM MXINT4 MoE")
    print("  Marlin path: grouped_topk (routing) + fused_marlin_moe (computation)")
    print("  TRT-LLM path: trtllm_mxint4_block_scale_moe (fused routing + computation)")
    print("=" * 80)
    print(f"Intermediate size (n): {args.intermediate_size}")
    print(f"Hidden size (k): {args.hidden_size}")
    print(f"Number of experts (e): {args.num_experts}")
    print(f"Top-K: {args.topk}")
    print(f"Group size: {args.group_size}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Benchmark iterations: {args.num_iters}")
    print("=" * 80)

    results = []
    for m in args.num_tokens:
        print(f"\nBenchmarking with num_tokens={m}...")
        result = run_benchmark(
            m=m,
            n=args.intermediate_size,
            k=args.hidden_size,
            e=args.num_experts,
            topk=args.topk,
            group_size=args.group_size,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )
        results.append(result)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(
        f"{'Tokens':>8} | {'Marlin (ms)':>12} | {'TRT-LLM (ms)':>13} | "
        f"{'Speedup':>8} | {'Max Diff':>10} | {'Mean Diff':>10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r['m']:>8} | {r['marlin_latency_ms']:>12.4f} | "
            f"{r['trtllm_latency_ms']:>13.4f} | {r['speedup']:>8.2f}x | "
            f"{r['max_diff']:>10.6f} | {r['mean_diff']:>10.6f}"
        )

    print("=" * 80)

    # Save to CSV if requested
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {args.csv}")


if __name__ == "__main__":
    main()
