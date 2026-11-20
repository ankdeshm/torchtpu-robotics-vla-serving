# Working Log — Robotics & VLA Serving on TorchTPU

## Goal
Bring up and benchmark Vision-Language-Action (VLA) robotics models on Google Cloud TPU v7x and TPU v5p using PyTorch / TorchTPU (`torch.compile(backend="tpu")`, `vllm-omni 0.33.0rc1`, and `sglang-torchtpu`), targeting real-time `20–50 Hz` robot control loops.

## What We Tried & Found

### 1. `GR00T-N1.7-3B` Bring-Up & Numerical Stability
- **Issue 1 (Cross-attention dimension):** Even-indexed DiT blocks in `GR00T-N1.7-3B` cross-attend to `2048`-dim VLM features while odd-indexed blocks self-attend on `1536`-dim action tokens. Passing the full `2048`-dim VLM output to even blocks resolved the shape mismatch.
- **Issue 2 (`bf16` activation overflow):** Early runs produced non-finite activations (`NaN`) through the 4-step DiT loop. Adding pre-attention and pre-FFN `LayerNorm` restored finite outputs across all camera counts and batch sizes.
- **Result:** Switching from eager (`65.2 ms` / `15.4 Hz`) to `torch.compile(backend="tpu")` cut latency `10.3×` on 1 TPU v7x core to `6.34 ms` (`p95 = 6.37 ms`, `157.7 Hz`) and `17.09 ms` (`58.5 Hz`, `234.1 chunks/s`) at `B=4`.

### 2. `Cosmos3-Edge-Policy-DROID` Bring-Up & `vllm-omni` Audit
- Compiled the MoT action tower with `torch.compile(backend="tpu")`, cutting `B=1` latency from `170.61 ms` eager to `6.57 ms` (`152.1 Hz`, `26.0×`) on 1 TPU v7x core and from `363.53 ms` eager to `9.40 ms` (`106.4 Hz`, `38.7×`) on 1 TPU v5p chip.
- On `vllm-omni 0.33.0rc1` (`deploy/cosmos3_policy_droid.yaml`), the server boots cleanly in `28–36 s`, but end-to-end WebSocket serving requires installing the external `cosmos_framework` transform package inside the server container.

### 3. `OpenVLA-7B` Compatibility on `transformers 5.19+`
- **Issues:** Loading `openvla/openvla-7b` on `transformers 5.19+` failed on `timm` version checks, removed `transformers.tokenization_utils` exports, missing `GenerationMixin` inheritance, and `tie_weights` signature changes.
- **Resolution:** Pinned `timm==0.9.16`, added a lightweight `transformers.tokenization_utils` shim after importing `AutoProcessor`, mixed in `GenerationMixin` with `_supports_sdpa = True`, and wrapped `tie_weights`. Achieved `373.09 ms` (`2.68 Hz`) eager 7-DoF `predict_action` and `11.28 ms` (`88.6 Hz`) compiled 1-step forward on 1 TPU v7x core (`19.37 ms` / `51.6 Hz` on TPU v5p). Note that `openvla-7b-oft-finetuned-libero-spatial` additionally requires `pip install prismatic`.

### 4. `Pi0.5` (`lerobot/pi05_base`) End-to-End WebSocket Bring-Up on `vllm-omni 0.33.0rc1`
- **Issue 1 (Gated backbone tokenizer):** `lerobot/pi05_base` fetches `google/paligemma-3b-pt-224` tokenizer assets; exporting `HF_TOKEN` with Gemma access resolved startup cleanly (`28–30 s` boot on TPU v7x).
- **Issue 2 (WebSocket state key):** Sending `"observation.state"` returned `RuntimeError: π0.5 requires a state`. Inspecting `processor_pi05.py` showed it reads `robot_obs.get("state")` (alongside `observation.images.base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb`).
- **Resolution:** Passing `"state"` (`float32[32]`) unlocked end-to-end `[50, 32]` 10-step action chunks at `215.66 ms` p50 (`p95 = 226.38 ms`, `4.64 Hz`, `231.8 actions/sec`) in `bfloat16` compiled mode and `215.24 ms` (`232.3 actions/sec`) in `float32` eager mode on TPU v7x.
