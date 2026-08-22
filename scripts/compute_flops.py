#!/usr/bin/env python
"""
Compute FLOPs and Parameters

Analyze model complexity for paper reporting.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from src.models.unified_hrm import UnifiedCVHRM, UnifiedCVHRMConfig
from src.models.hrm import HierarchicalVisionTransformer, HRMConfig
from src.utils.metrics import compute_flops, count_parameters, model_summary


def main():
    parser = argparse.ArgumentParser(description="Compute model FLOPs and parameters")
    parser.add_argument("--model", type=str, help="Path to model checkpoint or HF repo")
    parser.add_argument("--config", type=str, help="Path to model config YAML")
    parser.add_argument("--preset", type=str,
                        choices=["standard", "dsa", "moe", "dsa_moe", "paper_base"],
                        default="paper_base", help="Model preset")
    parser.add_argument("--input-size", type=int, nargs=4, default=[1, 3, 224, 224],
                        help="Input shape (B, C, H, W)")
    parser.add_argument("--unit", choices=["G", "M"], default="G",
                        help="FLOPs unit: G=GFLOPs, M=MFLOPs")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    # Build or load model
    if args.model:
        if Path(args.model).exists():
            # Local checkpoint
            checkpoint = torch.load(args.model, map_location="cpu")
            if "model_config" in checkpoint:
                model_cfg = HRMConfig.from_dict(checkpoint["model_config"])
            elif "config" in checkpoint:
                model_cfg = HRMConfig.from_dict(checkpoint["config"])
            else:
                print("No model config in checkpoint, using preset")
                model_cfg = getattr(HRMConfig, args.preset)()
            model = HierarchicalVisionTransformer(model_cfg)
            model.load_state_dict(
                checkpoint.get("model_state_dict", checkpoint), strict=False
            )
        else:
            # HF repo
            hf_config = UnifiedCVHRMConfig.from_pretrained(args.model)
            model = UnifiedCVHRM.from_pretrained(args.model, config=hf_config).backbone
    elif args.config:
        from src.config.model_config import HRMConfig
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_cfg = HRMConfig.from_dict(cfg.get("model", {}))
        model = HierarchicalVisionTransformer(model_cfg)
    else:
        model_cfg = getattr(HRMConfig, args.preset)()
        model = HierarchicalVisionTransformer(model_cfg)

    # Compute metrics
    print(model_summary(model, tuple(args.input_size)))

    flops_info = compute_flops(model, tuple(args.input_size), unit=args.unit)
    param_info = count_parameters(model)

    print(f"\nParameters: {param_info['total']:,} ({param_info['total_mb']:.2f} MB)")
    print(f"Trainable:  {param_info['trainable']:,}")
    print(f"FLOPs:      {flops_info['flops']:.2f} {flops_info['flops_unit']}FLOPs")

    # Save if requested
    if args.output:
        import json
        out = {
            "parameters": param_info,
            "flops": flops_info,
            "input_shape": args.input_size,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()