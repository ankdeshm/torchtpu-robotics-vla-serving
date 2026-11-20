# Cosmos3-Edge-Policy-DROID on Google Cloud TPUs (TorchTPU)

- **Model:** [`nvidia/Cosmos3-Edge-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID) (Mixture-of-Transformers VLA action policy for DROID manipulation, 537 action-tower tensors in `bf16`)
- **License:** NVIDIA Open Model License Agreement
- **Execution Stack:** PyTorch `2.13.0` + `torch_tpu 0.2.0` (`eager` vs. `torch.compile(backend="tpu")`, 1 TPU core / chip), plus `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`deploy/cosmos3_policy_droid.yaml`) boot audit

## Measured Action Tower Results (`4` Denoising Steps, `finite=True`)

| TPU Hardware (1 Device) | Execution Mode | Context Len | Action Horizon | Batch | Latency p50 | Latency p95 | Control Frequency | Aggregate Throughput | Compile Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 512 | 16 | 1 | **6.57 ms** | **6.60 ms** | **152.1 Hz** | 152.1 chunks/s | **26.0×** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 512 | 16 | 4 | **14.91 ms** | 15.56 ms | **67.1 Hz** | **268.3 chunks/s** | **9.6×** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 1024 | 16 | 1 | **7.35 ms** | 7.38 ms | **136.1 Hz** | 136.1 chunks/s | **16.0×** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 1024 | 32 | 1 | **7.71 ms** | 7.81 ms | **129.7 Hz** | 129.7 chunks/s | **15.5×** |
| TPU v7x (1 core) | Eager (`torch_tpu`) | 512 | 16 | 1 | 170.61 ms | 175.20 ms | 5.9 Hz | 5.9 chunks/s | 1.0× |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 512 | 16 | 1 | **9.40 ms** | **9.45 ms** | **106.4 Hz** | 106.4 chunks/s | **38.7×** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 512 | 16 | 4 | **15.12 ms** | 15.15 ms | **66.2 Hz** | **264.6 chunks/s** | **24.4×** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 1024 | 16 | 1 | **10.70 ms** | 10.76 ms | **93.4 Hz** | 93.4 chunks/s | **22.1×** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 1024 | 32 | 1 | **11.15 ms** | 11.27 ms | **89.7 Hz** | 89.7 chunks/s | **28.2×** |
| TPU v5p (1 chip) | Eager (`torch_tpu`) | 512 | 16 | 1 | 363.53 ms | 370.74 ms | 2.8 Hz | 2.8 chunks/s | 1.0× |

### Key Takeaways & `vllm-omni` Server Integration Audit
- **Up to 38.7× speedup from `torch.compile(backend="tpu")`:** Compiling the 537 MoT action-tower tensors drops `B=1` latency from `170.61 ms -> 6.57 ms` (**`152.1 Hz`**, `1.43×` faster than TPU v5p) on 1 TPU v7x core and from `363.53 ms -> 9.40 ms` (**`106.4 Hz`**) on 1 TPU v5p chip.
- **Doubling context length (`512 -> 1024`) and action horizon (`16 -> 32`) adds only `1.14 ms` on TPU v7x** (`6.57 ms -> 7.71 ms`, still **`129.7 Hz`**).
- **`vllm-omni 0.33.0rc1` native server audit:** Launching `vllm serve .../Cosmos3-Edge-Policy-DROID --omni --deploy-config .../vllm_omni/deploy/cosmos3_policy_droid.yaml` boots cleanly to `/health 200 OK` in `28 s` on TPU v7x and `36 s` on TPU v5p, but end-to-end WebSocket policy requests require the external `cosmos_framework` package to be installed inside the container for DROID observation preprocessing transforms.
