---
name: accuracy-debugging-vllm
description: Debug inference model accuracy and numerical correctness in vLLM/pm-vllm. Use when investigating wrong outputs, numerical mismatches vs baseline, kernel/layer accuracy, quantization or checkpoint issues, or when the user mentions accuracy debugging, minimum reproducer, or inference correctness.
---

# Inference Accuracy Debugging (vLLM)

Workflow for finding and fixing accuracy issues in inference. Issues are often multi-factor; use a systematic approach and a solid baseline.

## When to use this skill

- Outputs differ from a reference (HF, SGL, another backend).
- Numerical mismatches after kernel/layer/quantization changes.
- Suspected checkpoint, config, or model-parallelism issues.
- User asks for accuracy debugging, minimum reproducer, or inference correctness.

## Step 1 – Minimum reproducer and baseline

1. **Minimum reproducer**
   - Inference accuracy needs the full model; you cannot shrink layers like in perf debugging.
   - Vary and document: batch size, input/output sequence lengths, chunk-prefill size, max tokens per batch.
   - Narrow to the smallest config that still shows the bug.

2. **Baseline**
   - Define one trusted reference (e.g., HF transformers, another framework, or a known-good vLLM config).
   - Compare outputs (logits or chosen tokens) against this baseline.
   - Without a clear baseline, debugging degrades into trial and error.

3. **Multi-factor**
   - Assume multiple factors can contribute. Use the baseline to separate expected vs anomalous behavior.

## Step 2 – Locate the issue

Use the categories below to narrow the cause. For full checks and examples, see [reference.md](reference.md).

### Model checkpoints

- Missing or wrong scaling / post-processing of weights in specific modules.
- Wrong model or server config (e.g., inferred from config).
- Cross-check with the same model on another codebase (e.g., SGL/HF) when possible.

### Kernel

- **Streams**: Wrong stream or hidden cross-stream dependencies. Try `torch.cuda.synchronize()` before the suspicious kernel or `CUDA_LAUNCH_BLOCKING=1`. If accuracy fixes, fix stream usage.
- **Memory**: Uninitialized output buffers, too-small workspace, or misuse of sub-storage/pointers (especially with comms). Ensure correct buffer size and layout.
- **Numerics**: fast-math / underflow/overflow (e.g., CUTLASS). Compare compiler flags and build options with the baseline kernel.
- **Inputs**: Attention metadata (max seq length, cumulative lengths for Q/K/V, padding, block tables, CUDA graph max length). Stride/layout and reshape/transpose must match kernel expectations.
- **Environment**: If needed, lock GPU clock (e.g. `nvidia-smi -i 0 -lgc 1650`) to rule out frequency-related non-determinism.

### Algorithm

- Differences in chunked-prefill, prefill/decode scheduling, or quantization path.
- Use `torch.assert()` (or equivalent) at layer/function boundaries to isolate the first place values diverge.
- For quantized or rescaled ops (e.g., DeepGEMM), ensure scaling factors and backend choices match the intended algorithm.

### Model parallelism

- Reordering of all-reduce, matmul, softmax across devices can change results, especially in FP8/FP4.
- Batch-sensitive layers (e.g., LayerNorm, batch norm) can diverge if batch is split across devices.
- Use deterministic inference if the framework provides it; compare TP vs non-TP or different TP sizes to confirm.

## Workflow checklist

Copy and track:

```
Accuracy debugging progress:
- [ ] Step 1: Minimum reproducer (batch, seq lengths, chunk-prefill, tokens/batch)
- [ ] Step 1: Baseline defined and comparison method (logits/tokens, metric)
- [ ] Step 2: Checkpoints/config (scaling, post-processing, config vs reference)
- [ ] Step 2: Kernel (streams, memory, numerics, attention metadata, layout)
- [ ] Step 2: Algorithm (chunked-prefill, scheduling, quantization, asserts)
- [ ] Step 2: Model parallelism (reordering, batch stats, deterministic mode)
```

## Output format

When reporting findings:

1. **Reproducer**: Exact command/config and environment (vLLM version, GPU, dtype, quantization).
2. **Baseline**: What was compared (model, framework, config).
3. **Location**: Category (checkpoint/kernel/algorithm/TP) and, if known, file/layer or kernel name.
4. **Evidence**: What changed when (e.g., sync, CUDA_LAUNCH_BLOCKING, assert, or config change) and outcome.

## Project conventions

- Put debug notes and findings in `cs-scripts/` (e.g. `debug_<topic>_<date>.md`).
- Use `/workspace/logs` for logs, dumps, and traces.
- For pm-vllm source changes, announce and get confirmation before modifying; follow the git workflow rules.
