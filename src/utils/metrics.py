"""
Metrics Utilities

Helper metrics and computation tools.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from collections import defaultdict
import time


class AverageMeter:
    """Track average and current value of a metric."""

    def __init__(self, name: str = "", fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"


class MetricTracker:
    """Track multiple metrics with moving averages."""

    def __init__(self, *names: str):
        self.meters = {name: AverageMeter(name) for name in names}

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, name: str, val: float, n: int = 1):
        if name in self.meters:
            self.meters[name].update(val, n)

    def update_all(self, metrics: Dict[str, float], n: int = 1):
        for name, val in metrics.items():
            self.update(name, val, n)

    def avg(self, name: str) -> float:
        return self.meters[name].avg

    def result(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}

    def __str__(self):
        return " | ".join(str(m) for m in self.meters.values())


# ─────────────────────────────────────────────────────────────────
# Model Analysis
# ─────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, int]:
    """
    Count model parameters.

    Returns:
        Dict with 'total', 'trainable', 'non_trainable' counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total_mb": total * 4 / (1024 * 1024),  # FP32
    }


def compute_flops(
    model: nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    unit: str = "G",
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute FLOPs using fvcore (if available) or thop.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        unit: "G" for GFLOPs, "M" for MFLOPs
        device: Device to run on

    Returns:
        Dict with flops and params
    """
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        import warnings
        warnings.filterwarnings("ignore")

        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        dummy_input = torch.randn(*input_shape, device=device)
        flops = FlopCountAnalysis(model, dummy_input)

        total_flops = flops.total()
        params = sum(p.numel() for p in model.parameters())

        if unit == "G":
            total_flops = total_flops / 1e9
        elif unit == "M":
            total_flops = total_flops / 1e6

        return {
            "flops": total_flops,
            "params": params,
            "flops_unit": unit,
        }
    except ImportError:
        try:
            from thop import profile
            dummy_input = torch.randn(*input_shape)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)

            if unit == "G":
                flops = flops / 1e9
            elif unit == "M":
                flops = flops / 1e6

            return {"flops": flops, "params": params, "flops_unit": unit}
        except ImportError:
            return {"flops": -1, "params": -1, "error": "fvcore or thop not installed"}


def model_summary(model: nn.Module, input_shape: tuple = (1, 3, 224, 224)) -> str:
    """Generate model summary string."""
    params = count_parameters(model)
    flops = compute_flops(model, input_shape)

    lines = [
        "=" * 60,
        f"Model: {model.__class__.__name__}",
        "=" * 60,
        f"Total Parameters: {params['total']:,} ({params['total_mb']:.2f} MB)",
        f"Trainable: {params['trainable']:,}",
        f"Non-trainable: {params['non_trainable']:,}",
        "-" * 60,
    ]

    if flops.get("flops", -1) > 0:
        lines.append(f"FLOPs: {flops['flops']:.2f} {flops['flops_unit']}FLOPs")

    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────

class Timer:
    """Simple timer for profiling."""

    def __init__(self):
        self.times = defaultdict(list)
        self.start_times = {}

    def start(self, name: str):
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if name not in self.start_times:
            return 0.0
        elapsed = time.perf_counter() - self.start_times.pop(name)
        self.times[name].append(elapsed)
        return elapsed

    def avg(self, name: str) -> float:
        if name in self.times and self.times[name]:
            return sum(self.times[name]) / len(self.times[name])
        return 0.0

    def total(self, name: str) -> float:
        if name in self.times:
            return sum(self.times[name])
        return 0.0

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "avg": sum(t) / len(t),
                "total": sum(t),
                "count": len(t),
                "min": min(t),
                "max": max(t),
            }
            for name, t in self.times.items() if t
        }


def benchmark_model(
    model: nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark model inference time.

    Returns:
        Dict with mean/std latency, throughput
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    times = torch.tensor(times)
    mean_time = times.mean().item()
    std_time = times.std().item()

    throughput = input_shape[0] / mean_time  # samples/sec

    return {
        "mean_latency_ms": mean_time * 1000,
        "std_latency_ms": std_time * 1000,
        "throughput": throughput,
        "batch_size": input_shape[0],
    }