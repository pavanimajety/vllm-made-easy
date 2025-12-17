export VLLM_USE_NCCL_SYMM_MEM=1
export NCCL_NVLS_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1
export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA
export VLLM_FLASHINFER_MOE_BACKEND=latency
export VLLM_USE_FLASHINFER_MOE_FP4=1

vllm serve /tmp/nvidia-DeepSeek-R1-FP4-v2  --kv-cache-dtype fp8 --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
          --swap-space 16 --max-num-seqs 1024 --trust-remote-code --max-model-len 2176 --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192 --no-enable-prefix-caching \
          --async-scheduling --compilation_config.pass_config.enable_fi_allreduce_fusion true --compilation_config.pass_config.enable_attn_fusion true \
          --compilation_config.max_cudagraph_capture_size 2048


#------------ Works ------------------------------------------
# export VLLM_USE_NCCL_SYMM_MEM=1
# export NCCL_NVLS_ENABLE=1
# export NCCL_CUMEM_ENABLE=1
# export VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1
# export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA
# export VLLM_FLASHINFER_MOE_BACKEND=latency
# export VLLM_USE_FLASHINFER_MOE_FP4=1

# vllm serve /tmp/nvidia-DeepSeek-R1-0528-FP4/  --kv-cache-dtype fp8 --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
#           --swap-space 16 --max-num-seqs 1024 --trust-remote-code --max-model-len 2176 --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192 --no-enable-prefix-caching \
#           --async-scheduling --compilation_config.pass_config.enable_fi_allreduce_fusion true --compilation_config.pass_config.enable_attn_fusion true \
#           --compilation_config.max_cudagraph_capture_size 2048