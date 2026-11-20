# GR00T-N1.7-3B on Google Cloud TPUs (TorchTPU)

- **Model:** [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) (3B-parameter Vision-Language-Action model: 16-layer Eagle VLM + 4-step Flow-Matching DiT action head, `bf16`)
- **License:** NVIDIA Open Model License Agreement
- **Execution Stack:** PyTorch `2.13.0` + `torch_tpu 0.2.0` (`eager` vs. `torch.compile(backend="tpu")`, 1 TPU core / chip)

## Measured Control-Loop Results (`finite=True` on All Runs)

Measurements use `nvidia/GR00T-N1.7-3B` with pre-attention `LayerNorm` stabilization ensuring finite action chunks across all camera counts and batch sizes.

| TPU Hardware (1 Device) | Execution Mode | Cameras | Batch | Denoise Steps | Latency p50 | Latency p95 | Control Frequency | Aggregate Throughput | Meets 50 Hz SLA? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 1 | 1 | 4 | **6.34 ms** | **6.37 ms** | **157.7 Hz** | 157.7 chunks/s | **Yes** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 2 | 1 | 4 | **6.39 ms** | 6.45 ms | **156.5 Hz** | 156.5 chunks/s | **Yes** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 4 | 1 | 4 | **8.04 ms** | 8.10 ms | **124.3 Hz** | 124.3 chunks/s | **Yes** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 1 | 4 | 4 | **17.09 ms** | 17.46 ms | **58.5 Hz** | **234.1 chunks/s** | **Yes (4 robots)** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 1 | 8 | 4 | 33.01 ms | 33.25 ms | 30.3 Hz | **242.4 chunks/s** | Meets 20 Hz SLA |
| TPU v7x (1 core) | Eager (`torch_tpu`) | 1 | 1 | 4 | 65.15 ms | 67.84 ms | 15.4 Hz | 15.4 chunks/s | No |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 1 | 1 | 4 | **8.42 ms** | **8.46 ms** | **118.8 Hz** | 118.8 chunks/s | **Yes** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 2 | 1 | 4 | **9.00 ms** | 9.18 ms | **111.1 Hz** | 111.1 chunks/s | **Yes** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 4 | 1 | 4 | **9.82 ms** | 9.90 ms | **101.8 Hz** | 101.8 chunks/s | **Yes** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 1 | 4 | 4 | **18.19 ms** | 18.40 ms | **55.0 Hz** | **219.9 chunks/s** | **Yes (4 robots)** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 1 | 8 | 4 | 30.08 ms | 30.14 ms | 33.3 Hz | **266.0 chunks/s** | Meets 20 Hz SLA |
| TPU v5p (1 chip) | Eager (`torch_tpu`) | 1 | 1 | 4 | 95.72 ms | 100.98 ms | 10.5 Hz | 10.5 chunks/s | No |

### Key Takeaways
- **10.3×–11.4× compilation speedup unlocks >100 Hz real-time closed-loop control:** Eager TorchTPU executes at `10.5–15.4 Hz` (`65.2–95.7 ms`), whereas `torch.compile(backend="tpu")` reaches **`157.7 Hz` (`6.34 ms`, `p95 = 6.37 ms`) on 1 TPU v7x core** and **`118.8 Hz` (`8.42 ms`, `p95 = 8.46 ms`) on 1 TPU v5p chip**.
- **Deterministic tail latency:** `p95` sits within `0.03–0.08 ms` of `p50` at `B=1` on both TPU generations, critical for hard real-time robot actuation loops.
- **Multi-robot serving on a single accelerator:** Even at `B=4` concurrent robots, both TPU v7x (`17.09 ms`, `58.5 Hz`, `234.1 chunks/s`) and TPU v5p (`18.19 ms`, `55.0 Hz`, `219.9 chunks/s`) stay comfortably above the **50 Hz (`<20 ms`)** real-time control SLA.
