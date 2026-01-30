#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from visionai_resnet import ResNetImageAnalyzer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze images with ResNet (torchvision pretrained or custom ResNet from ResNet.md)."
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Image paths and/or directories containing images.",
    )

    p.add_argument(
        "--backend",
        choices=["torchvision", "custom"],
        default="torchvision",
        help="Model backend. 'torchvision' uses ImageNet pretrained weights if available.",
    )
    p.add_argument(
        "--arch",
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="torchvision ResNet architecture (only used when --backend torchvision).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device string like 'cpu', 'cuda', 'cuda:0'. Default: auto.",
    )

    p.add_argument("--topk", type=int, default=5, help="Top-K predictions to return.")
    p.add_argument(
        "--feature",
        action="store_true",
        help="Also return pooled feature vector (avgpool output).",
    )
    p.add_argument(
        "--labels-file",
        default=None,
        help="Optional labels txt file (one label per line). Overrides built-in labels.",
    )

    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a custom model checkpoint (.pt/.pth). Used only with --backend custom.",
    )
    p.add_argument(
        "--custom-num-classes",
        type=int,
        default=10,
        help="Number of classes for custom ResNet head. Used only with --backend custom.",
    )

    p.add_argument(
        "--output",
        default=None,
        help="Write JSON to this file instead of stdout.",
    )

    return p.parse_args()


def _collect_image_inputs(inputs: List[str]) -> List[str]:
    out: List[str] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.append(str(p))
        else:
            out.append(str(p))
    return out


def main() -> int:
    args = _parse_args()

    analyzer = ResNetImageAnalyzer(
        backend=args.backend,
        arch=args.arch,
        device=args.device,
        labels_file=args.labels_file,
        checkpoint_path=args.checkpoint,
        custom_num_classes=args.custom_num_classes,
    )
    try:
        results = []
        for inp in _collect_image_inputs(args.inputs):
            p = Path(inp)
            if p.is_dir():
                results.extend(
                    analyzer.analyze_dir(str(p), topk=args.topk, return_feature=args.feature)
                )
            else:
                results.append(
                    analyzer.analyze_image(str(p), topk=args.topk, return_feature=args.feature)
                )
        payload = analyzer.to_json(results)
    finally:
        analyzer.close()

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

