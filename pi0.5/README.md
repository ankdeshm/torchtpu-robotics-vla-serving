# Pi0.5 (`lerobot/pi05_base`) on Google Cloud TPUs (TorchTPU)

- **Model:** [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) (3B-parameter PaliGemma-3B VLM + 300M flow-matching action expert outputting `[50, 32]` continuous action chunks over 10 denoising steps)
- **License:** Apache-2.0 (model weights); note that the upstream tokenizer / PaliGemma backbone dependency (`google/paligemma-3b-pt-224`) uses the **Gemma Terms of Use** and requires accepting gated Hugging Face access before downloading.
- **Engines Tested:** `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`deploy/pi05.yaml`, OpenPI msgpack WebSocket `/v1/realtime/robot/openpi`) and `sglang-torchtpu` (`main`, `Pi05Pipeline`)

## Measured End-to-End Serving Results (`vllm-omni 0.33.0rc1` on TPU v7x, `finite=True`)

All runs measured 30–40 steady-state WebSocket round-trips after 5 warmup calls, returning `[50, 32]` continuous action chunks (50 future control steps × 32 action dims) with `action_finite=True`.

### 1. Precision & Compile Mode Comparison (`10` Denoising Steps)

| Engine & Precision | Mode (`enforce_eager`) | Boot Time | Latency p50 | Latency p95 | Latency p99 | Chunk Frequency | Individual Actions/s | Action Shape & Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vllm-omni 0.33.0rc1` (`bfloat16`)** | **Compiled (`false`)** | 30.0 s | **215.66 ms** | **226.38 ms** | **232.04 ms** | **4.64 Hz** | **231.8 act/s** | `[50, 32]`, `std=0.197` |
| **`vllm-omni 0.33.0rc1` (`float32`)** | **Eager (`true`)** | 28.0 s | **215.24 ms** | 268.32 ms | 268.82 ms | **4.65 Hz** | **232.3 act/s** | `[50, 32]`, `std=0.157` |
| `vllm-omni 0.33.0rc1` (`bfloat16`) | Eager (`true`) | 30.0 s | 240.84 ms | 251.64 ms | 295.76 ms | 4.15 Hz | 207.6 act/s | `[50, 32]`, `std=0.171` |

### 2. Denoising-Step Frontier (`bfloat16` Compiled, 40 Timed WebSocket Calls per Point)

| Denoising Steps | Boot Time | Latency p50 | Latency p95 | Latency p99 | Chunk Frequency | Individual Actions/s | Speedup vs. 10-Step | Action Std |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2` steps** | 33.0 s | **94.24 ms** | **103.02 ms** | **107.59 ms** | **10.61 Hz** | **530.6 act/s** | **2.39×** *(breaks 10 Hz)* | `0.227` |
| **`4` steps** | 33.0 s | **130.73 ms** | **135.46 ms** | **137.58 ms** | **7.65 Hz** | **382.5 act/s** | **1.72×** | `0.193` |
| **`5` steps** | 33.0 s | **148.81 ms** | **159.26 ms** | **161.21 ms** | **6.72 Hz** | **336.0 act/s** | **1.51×** | `0.205` |
| `10` steps | 33.0 s | 225.34 ms | 244.63 ms | 246.63 ms | 4.44 Hz | 221.9 act/s | `1.00×` | `0.193` |

### Key Takeaways & Client Schema Notes
- **Linear denoising-step latency decomposition (`~16.4 ms/step` + `~61 ms` fixed overhead):** Sweeping `num_inference_steps` from 10 down to 2 scales end-to-end WebSocket `p50` cleanly from `225.3 ms` to **`94.24 ms` (`10.61 Hz`, `530.6 actions/sec`)** — showing `~16.4 ms` per flow-matching step on top of `~61 ms` fixed 3-camera preprocessing + PaliGemma VLM prefill + msgpack WebSocket round-trip.
- **Tightest tail latency with `bfloat16` compiled (`p99 = 232.0 ms`, only `16.4 ms` above `p50` at 10 steps):** Compiling in `bfloat16` trims `p95` by `41.9 ms` vs. `float32` eager (`226.4` vs `268.3 ms`).
- **Fast native boot on both TPU v7x and TPU v5p:** Boots cleanly to `/health 200 OK` in `28–33 s` on TPU v7x and `36 s` on TPU v5p using `vllm_omni/deploy/pi05.yaml`.
- **Exact msgpack WebSocket observation schema (`/v1/realtime/robot/openpi`):**
  - Camera keys: `observation.images.base_0_rgb`, `observation.images.left_wrist_0_rgb`, `observation.images.right_wrist_0_rgb` (`uint8[224, 224, 3]`)
  - Proprioception state key: **`state`** (`float32[32]`) — note that `processor_pi05.py` reads `robot_obs.get("state")` (not `"observation.state"`), raising `π0.5 requires a state` if `"state"` is omitted.
  - Checkpoint compatibility: `lerobot/pi05_base` loads cleanly; `lerobot/pi05_libero_finetuned_v044` fails checkpoint format validation (`UnsupportedCheckpointFormat`).

## Reproduce

```bash
export HF_TOKEN=<your_hf_token_with_paligemma_access>

# vLLM-Omni Pi0.5 server
vllm serve /path/to/lerobot/pi05_base \
  --omni \
  --deploy-config $(python3 -c 'import vllm_omni, os; print(os.path.join(os.path.dirname(vllm_omni.__file__), "deploy/pi05.yaml"))') \
  --served-model-name lerobot/pi05_base \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code
```
