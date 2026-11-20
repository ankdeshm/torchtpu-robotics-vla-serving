# OpenVLA-7B on Google Cloud TPUs (TorchTPU)

- **Model:** [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b) (7.541B-parameter Prismatic VLM + Llama-2 7B backbone outputting 7-DoF discretized robot actions, `bf16`)
- **License:** MIT
- **Execution Stack:** PyTorch `2.13.0` + `torch_tpu 0.2.0` (`eager` 7-DoF `predict_action` + `torch.compile(backend="tpu")` 1-step forward, 1 TPU core / chip)

## Measured Results (`7.541B` Params, `action_finite=True`)

| TPU Hardware (1 Device) | Execution Mode | Cold Compile | Latency p50 | Latency p95 | Control Frequency | v7x vs. v5p Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x (1 core)** | **Eager full 7-DoF `predict_action`** | 1,509.5 ms | **373.09 ms** | 377.17 ms | **2.68 Hz** | **1.92×** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")` 1-step forward** | 25,043.0 ms | **11.28 ms** | 13.21 ms | **88.64 Hz** | **1.72×** |
| TPU v5p (1 chip) | Eager full 7-DoF `predict_action` | 2,310.0 ms | 714.65 ms | 729.88 ms | 1.40 Hz | 1.00× |
| TPU v5p (1 chip) | `torch.compile(backend="tpu")` 1-step forward | 31,420.0 ms | **19.37 ms** | 19.42 ms | **51.63 Hz** | 1.00× |

### Key Takeaways & `transformers 5.x` Compatibility Notes
- **Meets the 50 Hz robotics SLA on both TPU generations when compiled:** A single compiled forward pass over all `7.541B` parameters executes in **`11.28 ms` (`88.64 Hz`) on 1 TPU v7x core** and **`19.37 ms` (`51.63 Hz`) on 1 TPU v5p chip**.
- **Autoregressive 7-DoF eager overhead:** Uncompiled 7-token autoregressive `predict_action()` runs at `373.1 ms` (`2.68 Hz`) on TPU v7x (`1.92×` faster than `714.7 ms` / `1.40 Hz` on TPU v5p), showing why parallel action-chunking architectures (`GR00T-N1.7-3B`, `Cosmos3-Edge-Policy-DROID`, `OpenVLA-OFT`) or compiled single-step decoding are essential for >50 Hz control loops.
- **Running `openvla/openvla-7b` cleanly on `transformers 5.19+`:**
  1. Pin `timm==0.9.16` (OpenVLA enforces `0.9.10 <= timm < 1.0.0`).
  2. Inject a lightweight `transformers.tokenization_utils` compatibility module (exporting `PaddingStrategy`, `TruncationStrategy`, `PreTokenizedInput`, `TextInput`) after importing `AutoProcessor`.
  3. Mix `GenerationMixin` into the dynamically loaded `OpenVLAForActionPrediction` class with `_supports_sdpa = True`.
  - Note: `openvla-7b-oft-finetuned-libero-spatial` additionally requires the external `prismatic` Python package (`pip install prismatic`).

## Reproduce

```bash
pip install --no-deps timm==0.9.16
python3 ../scripts/benchmark_robotics_vla.py \
  --model openvla --model-path /path/to/openvla/openvla-7b --iters 12
```
