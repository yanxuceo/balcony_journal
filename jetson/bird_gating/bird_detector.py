"""
Bird detector using YOLOv8n TensorRT engine.
Acts as the "gating" layer for the plant symbiote project:
quickly decides whether a frame contains a bird before
invoking the downstream VLM (Gemma 4 E2B) for richer analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time

from ultralytics import YOLO

import cv2
import numpy as np

import argparse

# COCO class id for bird
BIRD_CLASS_ID = 14


@dataclass
class BirdDetection:
    """Result of running the gating layer on one image."""
    has_bird: bool
    detections: list = field(default_factory=list)  # [(bbox_xyxy, confidence), ...]
    original_image_path: Optional[str] = None
    crop_image_paths: list = field(default_factory=list)
    annotated_image_path: Optional[str] = None
    inference_ms: float = 0.0


class BirdDetector:
    def __init__(
        self,
        engine_path: str = "yolov8n.engine",
        conf_threshold: float = 0.25,
        warmup: bool = True,
    ):
        """
        Load the TensorRT engine and optionally warm it up.

        Args:
            engine_path: path to the .engine file
            conf_threshold: minimum confidence to count as a detection
            warmup: run one dummy inference to absorb TRT context init cost
        """
        print(f"[BirdDetector] Loading engine: {engine_path}")
        t0 = time.time()
        self.model = YOLO(engine_path, task="detect")
        self.conf_threshold = conf_threshold
        load_ms = (time.time() - t0) * 1000
        print(f"[BirdDetector] Engine loaded in {load_ms:.1f} ms")

        if warmup:
            self._warmup()

    def _warmup(self):
        """
        First inference includes one-time GPU memory alloc + kernel setup.
        Run once with a dummy input so the first real call is at steady-state speed.
        """
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        t0 = time.time()
        _ = self.model.predict(
            dummy,
            conf=self.conf_threshold,
            classes=[BIRD_CLASS_ID],
            verbose=False,
        )
        warmup_ms = (time.time() - t0) * 1000
        print(f"[BirdDetector] Warmup inference: {warmup_ms:.1f} ms")


    def _save_crops(
        self,
        image: np.ndarray,
        detections: list,
        output_dir: Path,
        image_stem: str,
    ) -> list[str]:
        """
        Crop each detection bbox from the image and save as separate files.
        Returns list of saved crop paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        crop_paths = []

        for i, (bbox, conf) in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Clamp to image bounds
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_path = output_dir / f"{image_stem}_bird_{i:02d}_conf{conf:.2f}.jpg"
            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(str(crop_path))

        return crop_paths

    def _save_annotated(
        self,
        image: np.ndarray,
        detections: list,
        output_path: Path,
    ) -> str:
        """
        Draw bounding boxes + confidence labels on the image and save.
        Returns the saved path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = image.copy()

        for bbox, conf in detections:
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Green box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label with confidence
            label = f"bird {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 255, 0), -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

        cv2.imwrite(str(output_path), annotated)
        return str(output_path)



    def detect_single(
        self,
        image_path: str,
        save_crops: bool = False,
        save_annotated: bool = False,
        output_root: str = "test_outputs",
    ) -> BirdDetection:
        """
        Run bird detection on a single image file.

        Args:
            image_path: path to an image file
            save_crops: if True, save each detection as a cropped image
            save_annotated: if True, save the full image with bboxes drawn
            output_root: base dir for outputs (contains crops/, annotated/)

        Returns:
            BirdDetection with detections, crop paths, annotation path, timing.
        """
        image_path = str(Path(image_path).resolve())

        t0 = time.time()
        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            classes=[BIRD_CLASS_ID],
            verbose=False,
        )
        inference_ms = (time.time() - t0) * 1000

        result = results[0]
        detections = []
        for box in result.boxes:
            bbox_xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            detections.append((bbox_xyxy, conf))

        crop_paths = []
        annotated_path = None

        if (save_crops or save_annotated) and len(detections) > 0:
            # Load the image once, reuse for both outputs
            image = cv2.imread(image_path)
            image_stem = Path(image_path).stem
            output_root_path = Path(output_root)

            if save_crops:
                crop_paths = self._save_crops(
                    image, detections,
                    output_root_path / "crops",
                    image_stem,
                )

            if save_annotated:
                annotated_path = self._save_annotated(
                    image, detections,
                    output_root_path / "annotated" / f"{image_stem}_annotated.jpg",
                )

        return BirdDetection(
            has_bird=len(detections) > 0,
            detections=detections,
            original_image_path=image_path,
            crop_image_paths=crop_paths,
            annotated_image_path=annotated_path,
            inference_ms=inference_ms,
        )
    
    def detect_batch(
        self,
        image_dir: str,
        save_crops: bool = True,
        save_annotated: bool = True,
        output_root: str = "test_outputs",
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> list[BirdDetection]:
        """
        Run detection on every image in a directory.

        Args:
            image_dir: directory containing test images
            save_crops: save crops for images with detections
            save_annotated: save annotated versions
            output_root: base dir for outputs
            extensions: file extensions to consider as images

        Returns:
            List of BirdDetection results, one per image.
        """
        image_dir_path = Path(image_dir)
        if not image_dir_path.is_dir():
            raise ValueError(f"Not a directory: {image_dir}")

        # Gather all image files (case-insensitive extension match)
        image_files = sorted([
            p for p in image_dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ])

        if not image_files:
            print(f"[BirdDetector] No images found in {image_dir}")
            return []

        print(f"\n[BirdDetector] Batch processing {len(image_files)} images from {image_dir}")
        print("-" * 70)

        results = []
        total_inference_ms = 0.0
        n_with_bird = 0

        for idx, img_path in enumerate(image_files, start=1):
            result = self.detect_single(
                str(img_path),
                save_crops=save_crops,
                save_annotated=save_annotated,
                output_root=output_root,
            )
            results.append(result)
            total_inference_ms += result.inference_ms
            if result.has_bird:
                n_with_bird += 1

            # Per-image summary line
            status = "🐦" if result.has_bird else "❌"
            top_conf = max((c for _, c in result.detections), default=0.0)
            print(
                f"  [{idx:3d}/{len(image_files)}] {status} {img_path.name:40s} "
                f"birds={len(result.detections):2d}  "
                f"top_conf={top_conf:.2f}  "
                f"{result.inference_ms:6.1f}ms"
            )

        # Aggregate summary
        avg_ms = total_inference_ms / len(image_files)
        print("-" * 70)
        print(f"[BirdDetector] Batch summary:")
        print(f"  Total images:        {len(image_files)}")
        print(f"  With bird detected:  {n_with_bird}  ({100 * n_with_bird / len(image_files):.1f}%)")
        print(f"  Avg inference:       {avg_ms:.1f} ms")
        print(f"  Avg FPS:             {1000 / avg_ms:.1f}")

        return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO bird detector (gating layer for plant symbiote project)"
    )

    # Input selection: exactly one of --image or --dir
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str,
        help="Path to a single image file"
    )
    input_group.add_argument(
        "--dir", type=str,
        help="Path to a directory of images (batch mode)"
    )

    # Detection parameters
    parser.add_argument(
        "--engine", type=str, default="yolov8n.engine",
        help="Path to TensorRT engine file (default: yolov8n.engine)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)"
    )

    # Output options
    parser.add_argument(
        "--output-root", type=str, default="test_outputs",
        help="Directory for crops and annotated images (default: test_outputs)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Disable saving crops and annotated images"
    )

    args = parser.parse_args()

    detector = BirdDetector(
        engine_path=args.engine,
        conf_threshold=args.conf,
    )

    save_outputs = not args.no_save

    if args.image:
        print(f"\n[CLI] detect_single on: {args.image}")
        result = detector.detect_single(
            args.image,
            save_crops=save_outputs,
            save_annotated=save_outputs,
            output_root=args.output_root,
        )
        print(f"  has_bird:        {result.has_bird}")
        print(f"  detections:      {len(result.detections)}")
        for i, (bbox, conf) in enumerate(result.detections):
            print(f"    [{i}] bbox={[round(x, 1) for x in bbox]}  conf={conf:.3f}")
        print(f"  inference_ms:    {result.inference_ms:.1f}")
        if result.crop_image_paths:
            print(f"  crops:           {result.crop_image_paths}")
        if result.annotated_image_path:
            print(f"  annotated:       {result.annotated_image_path}")

    else:  # args.dir
        detector.detect_batch(
            image_dir=args.dir,
            save_crops=save_outputs,
            save_annotated=save_outputs,
            output_root=args.output_root,
        )