# Appendix — Robotics & VLA Ablations, Numerical Stability Fixes & Compatibility Notes

This appendix documents eager-vs-compiled optimization ladders, numerical stability fixes, `transformers 5.x` compatibility shims, and serving audits for Robotics/VLA models on TPU v7x and TPU v5p.

---

## 1. Numerical Stability & Compatibility Engineering

Every latency and control-frequency measurement in this repository runs on PyTorch / TorchTPU (`torch 2.13.0`, `torch_tpu 0.2.0`) and verifies `torch.isfinite(action_out).all() == True`.

### `GR00T-N1.7-3B` Bring-Up Fixes
1. **Cross-attention Conditioning Dimension (`2048` vs `1536`):** In `nvidia/GR00T-N1.7-3B`, even-indexed DiT blocks cross-attend to the `2048`-dim Eagle VLM hidden states while odd-indexed blocks self-attend on `1536`-dim action tokens. Passing the full `2048`-dim VLM output tensor to even blocks resolved dimension mismatches cleanly.
2. **Eliminating `bf16` Overflow (`finite=False -> finite=True`):** Applying pre-attention and pre-FFN `F.layer_norm` stabilized activation magnitudes through the 4-step Flow-Matching DiT loop, achieving `finite=True` across all 26 eager and compiled configurations on both TPU v7x and TPU v5p.

### `OpenVLA-7B` (`7.541B` Params) on `transformers 5.19+`
Loading `openvla/openvla-7b` on modern TorchTPU containers (`transformers 5.19+`) hit five compatibility hurdles that were resolved without altering model weights:
1. Pinned `timm==0.9.16` (OpenVLA enforces `0.9.10 <= timm < 1.0.0`).
2. Imported `AutoProcessor` first, then injected a lightweight `transformers.tokenization_utils` compatibility module exporting `PaddingStrategy`, `TruncationStrategy`, `PreTokenizedInput`, and `TextInput` (so `transformers`' lazy importer does not overwrite it).
3. Loaded `OpenVLAForActionPrediction` via `get_class_from_dynamic_module`, mixed in `GenerationMixin`, set `_supports_sdpa = True`, and wrapped `tie_weights(*args, **kwargs)` for API compatibility.
- *Limitation:* `openvla/openvla-7b-oft-finetuned-libero-spatial` requires the external `prismatic` Python package (`pip install prismatic`) and was not measured without it.

---

## 2. Full Eager vs. `torch.compile(backend="tpu")` Optimization Ladder

### `GR00T-N1.7-3B` (4-step DiT + 16-layer VLM, 1 TPU Device)

| Hardware | Config (`cams`, `batch`) | Eager p50 (Hz) | `tpu` Compile p50 / p95 (Hz) | Compile Speedup | Aggregate Chunks/s |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x** | `1-cam, B=1` | 65.15 ms (15.4 Hz) | **6.34 ms / 6.37 ms (157.7 Hz)** | **10.28×** | 157.7 |
| **TPU v7x** | `2-cam, B=1` | 64.32 ms (15.6 Hz) | **6.39 ms / 6.45 ms (156.5 Hz)** | **10.07×** | 156.5 |
| **TPU v7x** | `4-cam, B=1` | 50.00 ms (20.0 Hz) | **8.04 ms / 8.10 ms (124.3 Hz)** | **6.22×** | 124.3 |
| **TPU v7x** | `1-cam, B=4` | 64.12 ms (15.6 Hz) | **17.09 ms / 17.46 ms (58.5 Hz)** | **3.75×** | 234.1 |
| **TPU v7x** | `1-cam, B=8` | 65.41 ms (15.3 Hz) | 33.01 ms / 33.25 ms (30.3 Hz) | 1.98× | 242.4 |
| **TPU v5p** | `1-cam, B=1` | 95.72 ms (10.5 Hz) | **8.42 ms / 8.46 ms (118.8 Hz)** | **11.37×** | 118.8 |
| **TPU v5p** | `2-cam, B=1` | 89.72 ms (11.2 Hz) | **9.00 ms / 9.18 ms (111.1 Hz)** | **9.97×** | 111.1 |
| **TPU v5p** | `4-cam, B=1` | 89.75 ms (11.1 Hz) | **9.82 ms / 9.90 ms (101.8 Hz)** | **9.14×** | 101.8 |
| **TPU v5p** | `1-cam, B=4` | 81.60 ms (12.3 Hz) | **18.19 ms / 18.40 ms (55.0 Hz)** | **4.49×** | 219.9 |
| **TPU v5p** | `1-cam, B=8` | 79.70 ms (12.6 Hz) | 30.08 ms / 30.14 ms (33.3 Hz) | 2.65× | **266.0** |

### `Cosmos3-Edge-Policy-DROID` (537 MoT Action-Tower Tensors, 4 Steps, 1 TPU Device)

| Hardware | Config (`ctx`, `act`, `B`) | Eager p50 (Hz) | `tpu` Compile p50 / p95 (Hz) | Compile Speedup | Aggregate Chunks/s |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x** | `ctx=512, act=16, B=1` | 170.61 ms (5.9 Hz) | **6.57 ms / 6.60 ms (152.1 Hz)** | **25.97×** | 152.1 |
| **TPU v7x** | `ctx=512, act=16, B=4` | 143.47 ms (7.0 Hz) | **14.91 ms / 15.56 ms (67.1 Hz)** | **9.62×** | **268.3** |
| **TPU v7x** | `ctx=1024, act=16, B=1` | 117.75 ms (8.5 Hz) | **7.35 ms / 7.38 ms (136.1 Hz)** | **16.02×** | 136.1 |
| **TPU v7x** | `ctx=1024, act=32, B=1` | 119.18 ms (8.4 Hz) | **7.71 ms / 7.81 ms (129.7 Hz)** | **15.46×** | 129.7 |
| **TPU v5p** | `ctx=512, act=16, B=1` | 363.53 ms (2.8 Hz) | **9.40 ms / 9.45 ms (106.4 Hz)** | **38.67×** | 106.4 |
| **TPU v5p** | `ctx=512, act=16, B=4` | 368.22 ms (2.7 Hz) | **15.12 ms / 15.15 ms (66.2 Hz)** | **24.35×** | 264.6 |
| **TPU v5p** | `ctx=1024, act=16, B=1` | 236.84 ms (4.2 Hz) | **10.70 ms / 10.76 ms (93.4 Hz)** | **22.13×** | 93.4 |
| **TPU v5p** | `ctx=1024, act=32, B=1` | 314.12 ms (3.2 Hz) | **11.15 ms / 11.27 ms (89.7 Hz)** | **28.17×** | 89.7 |

---

## 3. Native Server Integration Audit (`vllm-omni 0.33.0rc1` & `sglang-torchtpu`)

- **`Pi0.5` (`lerobot/pi05_base`):** Boots cleanly to `/health 200 OK` on `vllm-omni 0.33.0rc1` (`deploy/pi05.yaml`) in `28–30 s` (TPU v7x) and `36 s` (TPU v5p) once `HF_TOKEN` is exported for gated `google/paligemma-3b-pt-224` tokenizer access. Clients must pass camera keys (`observation.images.base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`) and proprioception key `"state"` (`float32[32]`). `lerobot/pi05_libero_finetuned_v044` is rejected by the checkpoint loader (`UnsupportedCheckpointFormat`).
- **`Cosmos3-Edge-Policy-DROID`:** Boots cleanly on `vllm-omni 0.33.0rc1` (`deploy/cosmos3_policy_droid.yaml`) in `28 s` (v7x) / `36 s` (v5p); end-to-end WebSocket action serving requires installing the external `cosmos_framework` transform library inside the server image.
