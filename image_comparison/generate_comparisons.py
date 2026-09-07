"""Generate qualitative comparison images with the local CoCoNet and ISNet weights.

CoCoNet and ISNet solve different tasks:
  * CoCoNet fuses a visible/infrared MSRS pair into one grayscale image.
  * ISNet predicts a binary infrared-small-target mask on IRSTD-1k.

The script therefore produces task-correct panels instead of comparing the two
outputs as if they represented the same quantity.  It does not modify either
model repository or any source dataset image.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
THERMAL_ROOT = SCRIPT_DIR.parent
COCONET_ROOT = THERMAL_ROOT / "CoCoNet-main"
ISNET_ROOT = THERMAL_ROOT / "ISNet-master"
MSRS_ROOT = THERMAL_ROOT / "MSRS"
IRSTD_ROOT = ISNET_ROOT / "IRSTD-1k"

DEFAULT_COCONET_WEIGHT = COCONET_ROOT / "logs_main_gpu" / "latest.pth"
DEFAULT_ISNET_RUN = ISNET_ROOT / "result" / "2026-09-06-12-22-34_ISNet_1k_AsymBi"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MSRS/CoCoNet and IRSTD-1k/ISNet comparison panels."
    )
    parser.add_argument("--num", type=int, default=4, help="Images per dataset (default: 4)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device; auto uses CUDA when available",
    )
    parser.add_argument(
        "--coconet-weight", type=Path, default=DEFAULT_COCONET_WEIGHT,
        help="CoCoNet .pth checkpoint",
    )
    parser.add_argument(
        "--isnet-weight", type=Path, default=None,
        help="ISNet .pkl checkpoint; default selects the largest IoU filename",
    )
    parser.add_argument(
        "--isnet-run", type=Path, default=DEFAULT_ISNET_RUN,
        help="ISNet result run used for automatic checkpoint selection",
    )
    parser.add_argument("--msrs-root", type=Path, default=MSRS_ROOT)
    parser.add_argument("--irstd-root", type=Path, default=IRSTD_ROOT)
    parser.add_argument("--msrs-split", default="test", choices=("train", "test"))
    parser.add_argument("--threshold", type=float, default=0.5, help="ISNet mask threshold")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--msrs-names", nargs="*", default=None,
        help="Optional MSRS stems, e.g. 00004N 00055D",
    )
    parser.add_argument(
        "--irstd-names", nargs="*", default=None,
        help="Optional IRSTD-1k stems, e.g. XDU0 XDU100",
    )
    args = parser.parse_args()
    if args.num < 1:
        parser.error("--num must be at least 1")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def image_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not files:
        raise RuntimeError(f"No images found in: {directory}")
    return files


def evenly_spaced(items: Sequence[str], count: int) -> list[str]:
    if len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[index] for index in indices]


def select_msrs_names(common: Iterable[str], count: int) -> list[str]:
    """Choose a stable mixture of day (D) and night (N) MSRS samples."""
    names = sorted(common)
    day = [name for name in names if name.upper().endswith("D")]
    night = [name for name in names if name.upper().endswith("N")]
    day_count = count // 2
    night_count = count - day_count
    selected = evenly_spaced(night, night_count) + evenly_spaced(day, day_count)
    if len(selected) < count:
        selected.extend(name for name in names if name not in selected)
    return selected[:count]


def select_irstd_names(
    candidates: Iterable[str], labels: dict[str, Path], count: int
) -> list[str]:
    """Choose stable samples across the range of ground-truth target areas."""
    areas: list[tuple[int, str]] = []
    for name in sorted(candidates):
        if name not in labels:
            continue
        mask = np.asarray(Image.open(labels[name]).convert("L"), dtype=np.uint8)
        areas.append((int(np.count_nonzero(mask)), name))
    nonempty = [(area, name) for area, name in areas if area > 0]
    ranked = nonempty or areas
    return evenly_spaced([name for _, name in sorted(ranked)], count)


def require_names(requested: Sequence[str], available: Iterable[str], dataset: str) -> list[str]:
    available_set = set(available)
    missing = [name for name in requested if name not in available_set]
    if missing:
        raise FileNotFoundError(f"Unknown {dataset} sample name(s): {', '.join(missing)}")
    return list(requested)


def pil_gray_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    # Match CoCoNet-main/main.py exactly.  Its test path reads the image as
    # float32 before torchvision.ToTensor(), so the values remain in 0..255
    # (ToTensor only scales uint8 arrays).  The checkpoint's BatchNorm layers
    # are also used in training mode because the official test function does
    # not call model.eval().
    array = np.asarray(image.convert("L"), dtype=np.float32)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def coconet_visible_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    # Match CoCoNet-main/main.py: ToTensor on a float32 image followed by
    # GrayscaleTransform, which retains channel 0 for RGB visible images.
    array = np.asarray(image.convert("RGB"), dtype=np.float32)[..., 0]
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def load_coconet(weight: Path, device: torch.device) -> torch.nn.Module:
    weight = weight.resolve()
    if not weight.is_file():
        raise FileNotFoundError(f"CoCoNet checkpoint not found: {weight}")
    sys.path.insert(0, str(COCONET_ROOT))
    try:
        model_module = importlib.import_module("models.model")
        model = model_module.Unet_resize_conv(ablation="full")
    finally:
        sys.path.pop(0)
    checkpoint = torch.load(weight, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    # The official CoCoNet test function loads the checkpoint but does not
    # switch to eval mode.  Keep that behavior so generated images match the
    # repository's own results_main/results_finetune outputs.
    return model.to(device).train()


def find_best_isnet_weight(run_dir: Path) -> Path:
    checkpoint_dir = run_dir.resolve() / "checkpoint"
    candidates = list(checkpoint_dir.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No ISNet .pkl checkpoints found in: {checkpoint_dir}")

    def score(path: Path) -> tuple[float, float, float]:
        iou_match = re.search(r"IoU-(\d+(?:\.\d+)?)_nIoU-(\d+(?:\.\d+)?)", path.name)
        epoch_match = re.search(r"Epoch-\s*(\d+)", path.name)
        if not iou_match:
            return (-1.0, -1.0, -1.0)
        return (
            float(iou_match.group(1)),
            float(iou_match.group(2)),
            float(epoch_match.group(1)) if epoch_match else -1.0,
        )

    return max(candidates, key=score)


def load_isnet(weight: Path, device: torch.device) -> torch.nn.Module:
    weight = weight.resolve()
    if not weight.is_file():
        raise FileNotFoundError(f"ISNet checkpoint not found: {weight}")
    # CoCoNet and ISNet both have a top-level ``utils`` package.  Remove the
    # CoCoNet package from the import cache before unpickling the full ISNet
    # module, otherwise Python may resolve ISNet's ``utils.AttrDict`` to the
    # wrong repository.
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]
    sys.path.insert(0, str(ISNET_ROOT / "model"))
    try:
        importlib.import_module("ISNet")
        model = torch.load(weight, map_location=device)
    finally:
        sys.path.pop(0)
    return model.to(device).eval()


def isnet_input(image: Image.Image, device: torch.device) -> torch.Tensor:
    rgb = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def gradient_input(data: torch.Tensor) -> torch.Tensor:
    kernel_v = data.new_tensor(((0, -1, 0), (0, 0, 0), (0, 1, 0))).view(1, 1, 3, 3)
    kernel_h = data.new_tensor(((0, 0, 0), (-1, 0, 1), (0, 0, 0))).view(1, 1, 3, 3)
    channels = []
    for index in range(3):
        channel = data[:, index:index + 1]
        vertical = F.conv2d(channel, kernel_v, padding=1)
        horizontal = F.conv2d(channel, kernel_h, padding=1)
        channels.append(torch.sqrt(vertical.square() + horizontal.square() + 1e-6))
    return torch.cat(channels, dim=1)


def as_gray(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")


def mask_overlay(image: Image.Image, probability: np.ndarray, threshold: float) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize((512, 512)), dtype=np.float32)
    mask = probability >= threshold
    overlay = base.copy()
    overlay[mask] = 0.42 * overlay[mask] + 0.58 * np.array((255, 40, 40), dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def labeled_panel(images: Sequence[Image.Image], labels: Sequence[str]) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("Each panel image needs one label")
    target_height = min(image.height for image in images)
    normalized = []
    for image in images:
        width = round(image.width * target_height / image.height)
        normalized.append(image.convert("RGB").resize((width, target_height), Image.Resampling.LANCZOS))
    title_height = 42
    gap = 6
    canvas = Image.new(
        "RGB",
        (sum(image.width for image in normalized) + gap * (len(normalized) - 1), target_height + title_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(20)
    x = 0
    for image, label in zip(normalized, labels):
        canvas.paste(image, (x, title_height))
        box = draw.textbbox((0, 0), label, font=title_font)
        text_width = box[2] - box[0]
        draw.text((x + (image.width - text_width) / 2, 9), label, fill="black", font=title_font)
        x += image.width + gap
    return canvas


def make_overview(panels: Sequence[tuple[str, Image.Image]], output: Path) -> None:
    if not panels:
        return
    thumb_width = 1500
    prepared: list[tuple[str, Image.Image]] = []
    for title, panel in panels:
        height = round(panel.height * thumb_width / panel.width)
        prepared.append((title, panel.resize((thumb_width, height), Image.Resampling.LANCZOS)))
    heading_height = 44
    gap = 14
    total_height = sum(image.height + heading_height for _, image in prepared) + gap * (len(prepared) - 1)
    canvas = Image.new("RGB", (thumb_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    heading_font = font(22)
    y = 0
    for title, image in prepared:
        draw.text((10, y + 8), title, fill="black", font=heading_font)
        y += heading_height
        canvas.paste(image, (0, y))
        y += image.height + gap
    canvas.save(output)


def generate_msrs(
    model: torch.nn.Module,
    root: Path,
    split: str,
    names: Sequence[str] | None,
    count: int,
    output_dir: Path,
    device: torch.device,
) -> list[tuple[str, Image.Image]]:
    visible_files = image_files(root / split / "vi")
    infrared_files = image_files(root / split / "ir")
    common = sorted(set(visible_files) & set(infrared_files))
    selected = require_names(names, common, "MSRS") if names else select_msrs_names(common, count)
    output_dir.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, Image.Image]] = []
    with torch.inference_mode():
        for name in selected:
            visible = Image.open(visible_files[name]).convert("RGB")
            infrared = Image.open(infrared_files[name]).convert("L")
            visible_tensor = coconet_visible_tensor(visible, device)
            infrared_tensor = pil_gray_tensor(infrared, device)
            # This argument order matches CoCoNet-main/main.py.
            prediction = model(visible_tensor, infrared_tensor)
            fused = prediction.squeeze().detach().cpu().numpy() * 127.5 + 127.5
            fused_image = as_gray(fused)
            fused_image.save(output_dir / f"{name}_coconet_fused.png")
            panel = labeled_panel(
                (visible, infrared, fused_image),
                ("MSRS visible", "MSRS infrared", "CoCoNet fusion"),
            )
            panel.save(output_dir / f"{name}_comparison.png")
            panels.append((f"MSRS {name}", panel))
            print(f"[MSRS] {name}")
    return panels


def generate_irstd(
    model: torch.nn.Module,
    root: Path,
    names: Sequence[str] | None,
    count: int,
    threshold: float,
    output_dir: Path,
    device: torch.device,
) -> list[tuple[str, Image.Image]]:
    input_files = image_files(root / "IRSTD1k_Img")
    label_files = image_files(root / "IRSTD1k_Label")
    test_file = root / "test.txt"
    if test_file.is_file():
        test_names = [line.strip() for line in test_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        test_names = sorted(set(input_files) & set(label_files))
    available = [name for name in test_names if name in input_files and name in label_files]
    selected = (
        require_names(names, available, "IRSTD-1k test")
        if names else select_irstd_names(available, label_files, count)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, Image.Image]] = []
    with torch.inference_mode():
        for name in selected:
            infrared = Image.open(input_files[name]).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            ground_truth = Image.open(label_files[name]).convert("L").resize((512, 512), Image.Resampling.NEAREST)
            data = isnet_input(infrared, device)
            logits, _ = model(data, gradient_input(data))
            probability = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
            probability_image = as_gray(probability * 255.0)
            binary_image = as_gray((probability >= threshold).astype(np.uint8) * 255)
            overlay = mask_overlay(infrared, probability, threshold)
            probability_image.save(output_dir / f"{name}_isnet_probability.png")
            binary_image.save(output_dir / f"{name}_isnet_mask.png")
            overlay.save(output_dir / f"{name}_isnet_overlay.png")
            panel = labeled_panel(
                (infrared, ground_truth, probability_image, binary_image, overlay),
                ("IRSTD infrared", "Ground truth", "ISNet probability", "ISNet mask", "ISNet overlay"),
            )
            panel.save(output_dir / f"{name}_comparison.png")
            panels.append((f"IRSTD-1k {name}", panel))
            print(f"[IRSTD-1k] {name}")
    return panels


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    print(f"Loading CoCoNet: {args.coconet_weight.resolve()}")
    coconet = load_coconet(args.coconet_weight, device)
    msrs_panels = generate_msrs(
        coconet, args.msrs_root.resolve(), args.msrs_split, args.msrs_names,
        args.num, output_dir / "msrs", device,
    )
    del coconet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    isnet_weight = args.isnet_weight or find_best_isnet_weight(args.isnet_run)
    print(f"Loading ISNet: {isnet_weight.resolve()}")
    isnet = load_isnet(isnet_weight, device)
    irstd_panels = generate_irstd(
        isnet, args.irstd_root.resolve(), args.irstd_names, args.num,
        args.threshold, output_dir / "irstd_1k", device,
    )
    del isnet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    make_overview(msrs_panels + irstd_panels, output_dir / "comparison_overview.png")
    print(f"Done. Overview: {output_dir / 'comparison_overview.png'}")


if __name__ == "__main__":
    main()
