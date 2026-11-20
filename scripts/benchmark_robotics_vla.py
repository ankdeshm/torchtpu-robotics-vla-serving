#!/usr/bin/env python3
"""PyTorch / TorchTPU Robotics & Vision-Language-Action (VLA) benchmark harness.

Benchmarks Vision-Language-Action models on Google Cloud TPU v7x and TPU v5p using
standard PyTorch `torch.accelerator` APIs and `torch.compile(backend="tpu")`:
- nvidia/GR00T-N1.7-3B (4-step DiT + 16-layer VLM)
- nvidia/Cosmos3-Edge-Policy-DROID (537 MoT action-tower tensors)
- openvla/openvla-7b (7.541B, 7-DoF action generation)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from types import ModuleType
from typing import Any

from PIL import Image


def install_transformers_v5_openvla_compat() -> None:
    """Compatibility shim enabling openvla/openvla-7b dynamic modules on transformers >= 5.x."""
    from transformers import AutoProcessor  # noqa: F401
    from transformers.utils import PaddingStrategy, TruncationStrategy

    mod = ModuleType("transformers.tokenization_utils")
    mod.PaddingStrategy = PaddingStrategy  # type: ignore[attr-defined]
    mod.TruncationStrategy = TruncationStrategy  # type: ignore[attr-defined]
    mod.PreTokenizedInput = Any  # type: ignore[attr-defined]
    mod.TextInput = str  # type: ignore[attr-defined]
    sys.modules["transformers.tokenization_utils"] = mod


def benchmark_openvla(model_path: str, iters: int = 12) -> list[dict[str, Any]]:
    import torch
    import torch_tpu  # noqa: F401
    from transformers import AutoConfig, AutoProcessor, GenerationMixin
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    torch.accelerator.set_device_index(0)
    device = torch.device("tpu:0")
    install_transformers_v5_openvla_compat()

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cls_ref = cfg.auto_map["AutoModelForVision2Seq"]
    base_cls = get_class_from_dynamic_module(cls_ref, model_path)

    class OpenVLACompat(base_cls, GenerationMixin):
        _supports_sdpa = True

        def tie_weights(self, *args: Any, **kwargs: Any) -> Any:
            try:
                return super().tie_weights()
            except TypeError:
                return None

    model = OpenVLACompat.from_pretrained(model_path, config=cfg, dtype=torch.bfloat16, trust_remote_code=True).eval().to(device)
    load_s = time.perf_counter() - t0
    params_b = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)

    img = Image.new("RGB", (224, 224), color=(110, 140, 170))
    prompt = "In: What action should the robot take to pick up the cube?\nOut:"
    inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)

    results: list[dict[str, Any]] = []

    # 1) Full 7-DoF eager predict_action
    with torch.no_grad():
        torch.accelerator.synchronize()
        tc0 = time.perf_counter()
        act = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        torch.accelerator.synchronize()
        cold_ms = (time.perf_counter() - tc0) * 1000.0
        lats = []
        for _ in range(max(4, iters // 2)):
            ts = time.perf_counter()
            act = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            torch.accelerator.synchronize()
            lats.append((time.perf_counter() - ts) * 1000.0)
    p50 = statistics.median(lats)
    p95 = sorted(lats)[min(len(lats) - 1, int(0.95 * len(lats)))]
    results.append({
        "model": "openvla/openvla-7b",
        "mode": "eager_predict_action_7dof",
        "params_b": params_b,
        "load_s": round(load_s, 2),
        "cold_ms": round(cold_ms, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "control_hz_p50": round(1000.0 / p50, 2),
        "action_finite": bool(all(abs(float(v)) < 1e6 for v in act)),
    })

    # 2) torch.compile(backend="tpu") 1-step forward
    def fwd_1step(input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        return model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values).logits[:, -1, -7:]

    compiled = torch.compile(fwd_1step, backend="tpu")
    with torch.no_grad():
        torch.accelerator.synchronize()
        tc0 = time.perf_counter()
        out = compiled(inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"])
        torch.accelerator.synchronize()
        cold_ms = (time.perf_counter() - tc0) * 1000.0
        lats = []
        for _ in range(iters):
            ts = time.perf_counter()
            out = compiled(inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"])
            torch.accelerator.synchronize()
            lats.append((time.perf_counter() - ts) * 1000.0)
    p50 = statistics.median(lats)
    p95 = sorted(lats)[min(len(lats) - 1, int(0.95 * len(lats)))]
    results.append({
        "model": "openvla/openvla-7b",
        "mode": "torch.compile(backend='tpu')_fwd_1step",
        "params_b": params_b,
        "load_s": round(load_s, 2),
        "cold_ms": round(cold_ms, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "control_hz_p50": round(1000.0 / p50, 2),
        "action_finite": bool(torch.isfinite(out.float()).all().item()),
    })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Robotics/VLA models on TorchTPU.")
    parser.add_argument("--model", choices=["openvla"], default="openvla")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--iters", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(benchmark_openvla(args.model_path, args.iters), indent=2))


if __name__ == "__main__":
    main()
