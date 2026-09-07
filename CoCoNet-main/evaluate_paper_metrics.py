"""Evaluate infrared-visible fusion results with CoCoNet paper metrics.

The six metrics are exactly the metrics listed in Sec. 4.1.3 of
"CoCoNet: Coupled Contrastive Learning Network with Multi-level Feature
Ensemble for Multi-modality Image Fusion" (IJCV, 2024): EN, AG, SF, SD,
SCD, and VIF. Higher values are better for all six metrics.

Example (PowerShell, all matched TNO pairs):
    python .\evaluate_paper_metrics.py `
      --vis-dir .\dataset\TNO\VIS `
      --ir-dir .\dataset\TNO\IR `
      --fused-dir .\results_finetune `
      --per-image

For the 20-pair protocol used in the paper's Fig. 10, pass ``--limit 20``
or, preferably, ``--names-file`` containing the authors' exact filenames.

VIF follows the standard multi-scale pixel-domain VIF calculation cited by
the paper (Han et al., 2013). For fusion, it is the sum of source-to-fused
VIF for the visible and infrared source images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


EPS = np.finfo(np.float64).eps


def read_gray(path: Path) -> np.ndarray:
    """Load a grayscale image as float64, preserving its 0-255 intensities."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def entropy(image: np.ndarray) -> float:
    """EN, Eq. (8): Shannon entropy of the fused image."""
    values = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    probability = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probability /= probability.sum()
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def average_gradient(image: np.ndarray) -> float:
    """AG, Eq. (6): mean absolute horizontal plus vertical gradient."""
    horizontal = np.abs(np.diff(image, axis=1)).sum()
    vertical = np.abs(np.diff(image, axis=0)).sum()
    return float((horizontal + vertical) / image.size)


def spatial_frequency(image: np.ndarray) -> float:
    """SF, Eqs. (15)-(17): combined horizontal and vertical frequency."""
    height, width = image.shape
    horizontal = np.sqrt(np.sum(np.diff(image, axis=1) ** 2) / (height * width))
    vertical = np.sqrt(np.sum(np.diff(image, axis=0) ** 2) / (height * width))
    return float(np.sqrt(horizontal**2 + vertical**2))


def standard_deviation(image: np.ndarray) -> float:
    """SD, Eq. (18): population standard deviation of fused intensities."""
    return float(np.sqrt(np.mean((image - image.mean()) ** 2)))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / (denominator + EPS))


def scd(visible: np.ndarray, infrared: np.ndarray, fused: np.ndarray) -> float:
    """SCD, Eqs. (19)-(20): sum of correlations of source differences."""
    visible_difference = visible - fused
    infrared_difference = infrared - fused
    return correlation(visible, visible_difference) + correlation(infrared, infrared_difference)


def vif_single(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Four-scale pixel-domain VIF used by the paper's cited VIF metric."""
    reference = reference.astype(np.float64)
    distorted = distorted.astype(np.float64)
    numerator = 0.0
    denominator = 0.0
    sigma_nsq = 2.0

    for scale in range(1, 5):
        kernel_size = 2 ** (5 - scale) + 1
        sigma = kernel_size / 5.0
        if scale > 1:
            reference = cv2.GaussianBlur(reference, (kernel_size, kernel_size), sigma)[::2, ::2]
            distorted = cv2.GaussianBlur(distorted, (kernel_size, kernel_size), sigma)[::2, ::2]

        mu_ref = cv2.GaussianBlur(reference, (kernel_size, kernel_size), sigma)
        mu_dist = cv2.GaussianBlur(distorted, (kernel_size, kernel_size), sigma)
        ref_sq = mu_ref * mu_ref
        dist_sq = mu_dist * mu_dist
        ref_dist = mu_ref * mu_dist

        sigma_ref = cv2.GaussianBlur(reference * reference, (kernel_size, kernel_size), sigma) - ref_sq
        sigma_dist = cv2.GaussianBlur(distorted * distorted, (kernel_size, kernel_size), sigma) - dist_sq
        sigma_ref_dist = cv2.GaussianBlur(reference * distorted, (kernel_size, kernel_size), sigma) - ref_dist

        sigma_ref = np.maximum(sigma_ref, 0.0)
        sigma_dist = np.maximum(sigma_dist, 0.0)
        gain = sigma_ref_dist / (sigma_ref + EPS)
        residual = sigma_dist - gain * sigma_ref_dist

        # The following branches are the standard VIF stability conditions.
        mask_ref_small = sigma_ref < EPS
        gain[mask_ref_small] = 0.0
        residual[mask_ref_small] = sigma_dist[mask_ref_small]
        mask_dist_small = sigma_dist < EPS
        gain[mask_dist_small] = 0.0
        residual[mask_dist_small] = 0.0
        mask_gain_negative = gain < 0
        residual[mask_gain_negative] = sigma_dist[mask_gain_negative]
        gain[mask_gain_negative] = 0.0
        residual = np.maximum(residual, EPS)

        numerator += np.sum(np.log10(1.0 + (gain**2) * sigma_ref / (residual + sigma_nsq)))
        denominator += np.sum(np.log10(1.0 + sigma_ref / sigma_nsq))

    return float(numerator / (denominator + EPS))


def vif(visible: np.ndarray, infrared: np.ndarray, fused: np.ndarray) -> float:
    """VIF for fusion: visible-to-fused VIF plus infrared-to-fused VIF."""
    return vif_single(visible, fused) + vif_single(infrared, fused)


def resolve_images(directory: Path) -> dict[str, Path]:
    extensions = ("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = [path for pattern in extensions for path in directory.glob(pattern)]
    return {path.stem: path for path in files}


def main() -> None:
    parser = argparse.ArgumentParser(description="CoCoNet paper metric evaluator (EN, AG, SF, SD, SCD, VIF)")
    parser.add_argument("--vis-dir", type=Path, required=True, help="Visible source-image directory")
    parser.add_argument("--ir-dir", type=Path, required=True, help="Infrared source-image directory")
    parser.add_argument("--fused-dir", type=Path, required=True, help="Directory containing fused output images")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N matched names (paper Fig. 10 uses 20; exact list is not published)",
    )
    parser.add_argument(
        "--names-file", type=Path, default=None,
        help="Text file with exact image stems/names to evaluate, one per line; overrides --limit",
    )
    parser.add_argument("--per-image", action="store_true", help="Also print all individual-image metric values")
    args = parser.parse_args()

    visible_files = resolve_images(args.vis_dir)
    infrared_files = resolve_images(args.ir_dir)
    fused_files = resolve_images(args.fused_dir)
    names = sorted(set(visible_files) & set(infrared_files) & set(fused_files))
    if args.names_file is not None:
        requested = {
            line.strip().replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
            for line in args.names_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        names = [name for name in names if name in requested]
    elif args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        names = names[:args.limit]
    if not names:
        raise RuntimeError("No matched visible, infrared, and fused image filenames were found.")

    rows: list[tuple[str, float, float, float, float, float, float]] = []
    for name in names:
        visible = read_gray(visible_files[name])
        infrared = read_gray(infrared_files[name])
        fused = read_gray(fused_files[name])
        if visible.shape != infrared.shape or visible.shape != fused.shape:
            raise ValueError(
                f"Image dimensions must match for {name}: "
                f"VIS={visible.shape}, IR={infrared.shape}, fused={fused.shape}"
            )
        rows.append((
            name,
            entropy(fused),
            average_gradient(fused),
            spatial_frequency(fused),
            standard_deviation(fused),
            scd(visible, infrared, fused),
            vif(visible, infrared, fused),
        ))

    values = np.asarray([row[1:] for row in rows], dtype=np.float64)
    print(f"Matched image pairs: {len(rows)}")
    print("Paper metrics (mean +/- population std; higher is better):")
    labels = ("EN", "AG", "SF", "SD", "SCD", "VIF")
    for index, label in enumerate(labels):
        print(f"{label} = {values[:, index].mean():.6f} +/- {values[:, index].std(ddof=0):.6f}")

    if args.per_image:
        print("\nPer-image metrics:")
        print("image\tEN\tAG\tSF\tSD\tSCD\tVIF")
        for row in rows:
            print(row[0] + "\t" + "\t".join(f"{value:.6f}" for value in row[1:]))


if __name__ == "__main__":
    main()
