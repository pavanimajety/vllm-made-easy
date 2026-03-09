# Accuracy Debugging Reference (vLLM)

Condensed from the Inference Model Accuracy Debugging Workflow. Use for quick lookup during debugging.

## Step 1 – Reproducer and baseline

| Focus | Notes |
|-------|--------|
| Minimum reproducer | Full model required; vary batch size, input/output seq lengths, chunk-prefill size, tokens per batch. |
| Baseline | One trusted reference (e.g. HF, SGL). Compare logits or chosen tokens. |
| Multi-factor | Assume several causes; baseline separates expected vs anomalous. |

## Step 2 – Categories and checks

### Model checkpoints

- Missing or incorrect scaling / post-processing in certain modules → cross-platform mismatch.
- Incorrect model or server config inferred from config.
- References: SGL issues/PRs on scaling and config (see PDF for links).

### Kernel

| Cause | What to do |
|-------|------------|
| **Stream** | `torch.cuda.synchronize()` before kernel, or `CUDA_LAUNCH_BLOCKING=1`. If fix → fix stream usage. |
| **Memory** | Init output buffers; ensure workspace size; avoid wrong sub-storage/pointer use in comms. |
| **Underflow/overflow** | fast-math (e.g. gcc) can change subnormals/NaN/Inf. Align kernel and baseline compile flags. Ref: https://simonbyrne.github.io/notes/fastmath/ |
| **Attention metadata** | Max seq length, cumulative seq lengths for Q/K/V (easy to get wrong). Padded vs non-padded block tables; CUDA graph max length vs actual max length. |
| **Stride/layout** | Match kernel: reshape, view, transpose, layout swizzling. |
| **Environment** | Lock GPU clock if needed: `nvidia-smi -i 0 -lgc 1650`. |

### Algorithm

- Chunked-prefill, prefill/decode scheduling, or quantization path differences.
- Insert `torch.assert()` at layers/functions to find first divergence.
- DeepGEMM / scaling: wrong scaling factor or backend can cause large errors; check per-layer and per-backend.

### Model parallelism

- All-reduce / matmul / softmax order can change with TP; FP8/FP4 especially sensitive.
- Batch stats (LayerNorm, etc.) can differ when batch is split across devices.
- Use deterministic inference if available; compare TP vs non-TP.
- Ref: SGL TP deterministic work (see PDF for PR link).

## Useful commands

```bash
# Force synchronous CUDA (rule out stream bugs)
CUDA_LAUNCH_BLOCKING=1 python -m vllm.entrypoints.openai.api_server ...

# Lock GPU clock (example; adjust frequency to your GPU)
nvidia-smi -i 0 -lgc 1650
```

## Source

Workflow source: `cs-scripts/AccuracyDebuggingWorkflow.pdf` (Inference Model Accuracy Debugging Workflow).
