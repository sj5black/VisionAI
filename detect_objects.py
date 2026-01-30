#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from visionai_resnet.detector import ResNetObjectDetector


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect objects in images using ResNet-backbone detectors (Faster R-CNN / RetinaNet)."
    )
    p.add_argument("inputs", nargs="+", help="Image paths and/or directories containing images.")
    p.add_argument(
        "--model",
        default="fasterrcnn_resnet50_fpn_v2",
        choices=["fasterrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2", "fasterrcnn", "retinanet"],
        help="Detection model (both use ResNet50 backbone).",
    )
    p.add_argument("--device", default=None, help="Device string like 'cpu', 'cuda', 'cuda:0'. Default: auto.")
    p.add_argument("--threshold", type=float, default=0.5, help="Score threshold for keeping detections.")
    p.add_argument("--max-detections", type=int, default=100, help="Max detections per image (after threshold).")
    p.add_argument(
        "--labels-file",
        default=None,
        help="Optional labels txt file (one label per line). Overrides built-in labels.",
    )
    p.add_argument("--output", default=None, help="Write JSON to this file instead of stdout.")
    p.add_argument(
        "--save-vis",
        default=None,
        help=(
            "If set, save annotated images with boxes. "
            "Provide a directory path (recommended)."
        ),
    )
    return p.parse_args()


def _expand_inputs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    out.append(child)
        else:
            out.append(p)
    return out


def main() -> int:
    args = _parse_args()
    detector = ResNetObjectDetector(
        model_name=args.model,
        device=args.device,
        labels_file=args.labels_file,
    )

    results = []
    for img_path in _expand_inputs(args.inputs):
        r = detector.detect_image(
            str(img_path),
            score_threshold=args.threshold,
            max_detections=args.max_detections,
        )
        results.append(r)

        if args.save_vis:
            out_dir = Path(args.save_vis)
            out_path = out_dir / f"{img_path.stem}.detected{img_path.suffix}"
            detector.draw_boxes(str(img_path), r, output_path=str(out_path))

    payload = detector.to_json(results)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

