"""
Basic U-Net for extracting object contours from MSRS infrared images.

Expected MSRS layout:
    D:/twj/thermal/MSRS/
      train/ir/*.png
      train/Segmentation_labels/*.png
      test/ir/*.png
      test/Segmentation_labels/*.png

The original labels are multi-class segmentation labels. By default, all
non-zero classes are merged into foreground and converted to a slightly
dilated contour target. The dilation improves tolerance to small boundary
misalignments during training. Use --target mask to train foreground-mask
prediction instead.

Example:
    python unet_msrs.py --data-root D:/twj/thermal/MSRS --epochs 30
    python unet_msrs.py --data-root D:/twj/thermal/MSRS --target mask
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_contour(mask: np.ndarray) -> np.ndarray:
    """Return foreground boundary pixels using a 4-neighbour erosion."""
    fg = mask > 0
    eroded = fg.copy()
    eroded[1:, :] &= fg[:-1, :]
    eroded[:-1, :] &= fg[1:, :]
    eroded[:, 1:] &= fg[:, :-1]
    eroded[:, :-1] &= fg[:, 1:]
    return (fg & ~eroded).astype(np.float32)


def dilate_binary(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Morphologically dilate a binary mask with a 3x3 neighbourhood."""
    result = mask.astype(bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
            padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] |
            padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
        )
    return result.astype(np.float32)


class MSRSContourDataset(Dataset):
    def __init__(self, root: str | Path, split: str,
                 image_size: tuple[int, int] = (480, 640),
                 target: str = "contour", contour_width: int = 3) -> None:
        self.root = Path(root)
        self.split = split
        self.size = image_size  # (height, width), preserving MSRS 480x640 ratio
        self.target = target
        self.contour_width = max(1, contour_width)
        self.ir_dir = self.root / split / "ir"
        self.label_dir = self.root / split / "Segmentation_labels"
        if not self.ir_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {self.ir_dir} and {self.label_dir}. "
                "Check --data-root and MSRS directory names."
            )
        self.items = []
        for image_path in sorted(self.ir_dir.glob("*.png")):
            label_path = self.label_dir / image_path.name
            if label_path.exists():
                self.items.append((image_path, label_path))
        if not self.items:
            raise RuntimeError(f"No paired IR/label PNG files found in {self.root / split}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, label_path = self.items[index]
        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")
        image = image.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
        label = label.resize((self.size[1], self.size[0]), Image.Resampling.NEAREST)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        label_np = np.asarray(label, dtype=np.uint8)
        if self.target == "contour":
            contour = binary_contour(label_np)
            # width=1 keeps the original one-pixel contour; width=3 uses one
            # 3x3 dilation and gives roughly a 3-pixel boundary band.
            dilation_steps = (self.contour_width - 1) // 2
            target_np = dilate_binary(contour, dilation_steps)
        else:
            target_np = (label_np > 0).astype(np.float32)
        return (
            torch.from_numpy(image_np[None]),
            torch.from_numpy(target_np[None]),
            image_path.name,
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ROIAttention(nn.Module):
    """Learn a soft spatial ROI map and enhance, rather than erase, features.

    The sigmoid map has one value per spatial position.  The residual gate
    ``x * (1 + alpha * roi)`` keeps the original feature available even when
    the ROI predictor is uncertain during early training.
    """

    def __init__(self, channels: int, alpha: float = 1.0) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.predictor = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.alpha = float(alpha)

    def forward(self, x):
        roi = torch.sigmoid(self.predictor(x))
        return x * (1.0 + self.alpha * roi), roi


class UNet(nn.Module):
    def __init__(self, base_channels: int = 32, roi_alpha: float = 1.0) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(1, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.roi_attention = ROIAttention(c * 16, alpha=roi_alpha)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)
        self.out = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _cat(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x, return_roi: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        b, roi = self.roi_attention(b)
        d4 = self.dec4(self._cat(self.up4(b), e4))
        d3 = self.dec3(self._cat(self.up3(d4), e3))
        d2 = self.dec2(self._cat(self.up2(d3), e2))
        d1 = self.dec1(self._cat(self.up1(d2), e1))
        logits = self.out(d1)
        return (logits, roi) if return_roi else logits


def dice_loss(logits, targets, smooth: float = 1.0):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    score = (2 * intersection + smooth) / (probs.sum(dims) + targets.sum(dims) + smooth)
    return 1 - score.mean()


def contour_metrics(logits, targets, threshold: float = 0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * targets).sum().item()
    precision = inter / (pred.sum().item() + 1e-8)
    recall = inter / (targets.sum().item() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def run_epoch(model, loader, optimizer, device, train: bool, desc: str = "",
              scaler=None, use_amp: bool = False):
    model.train(train)
    total_loss = 0.0
    sums = np.zeros(3, dtype=np.float64)
    # tqdm reports batch-level progress and ETA. The outer epoch progress bar
    # in main() reports the estimated remaining time for the complete run.
    iterator = tqdm(
        loader,
        desc=desc,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.set_grad_enabled(train):
        for images, targets, _ in iterator:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                # Contours are sparse, so BCE+Dice is more stable than BCE alone.
                loss = F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * images.size(0)
            sums += np.asarray(contour_metrics(logits.detach(), targets)) * images.size(0)
            iterator.set_postfix(loss=f"{loss.item():.4f}")
    n = len(loader.dataset)
    return total_loss / n, *(sums / n)


@torch.no_grad()
def save_predictions(model, loader, device, output_dir: Path):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for images, _, names in loader:
        pred = (torch.sigmoid(model(images.to(device))) > 0.5).cpu().numpy()
        for mask, name in zip(pred[:, 0], names):
            Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / name)


def colorize_label(label: np.ndarray) -> Image.Image:
    """Make the MSRS multi-class label easier to inspect visually."""
    palette = np.array([
        [0, 0, 0], [220, 40, 40], [40, 180, 70], [40, 100, 220],
        [240, 180, 40], [180, 60, 220], [40, 200, 200], [220, 100, 40],
        [180, 180, 180],
    ], dtype=np.uint8)
    rgb = palette[np.clip(label.astype(np.int64), 0, len(palette) - 1)]
    return Image.fromarray(rgb, mode="RGB")


def save_contour_visualizations(model, dataset, device, output_dir: Path,
                                max_images: int = 20) -> None:
    """Save a five-panel explanation of the contour extraction process.

    Panels are: IR input | original multi-class label | contour target used for
    training | predicted contour | IR with predicted contour overlaid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = output_dir / "roi_attention"
    roi_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = min(max_images, len(dataset))
    with torch.no_grad():
        for index in range(count):
            image_tensor, target_tensor, name = dataset[index]
            logits, roi = model(image_tensor.unsqueeze(0).to(device), return_roi=True)
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
            roi_map = F.interpolate(
                roi, size=dataset.size, mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()
            image = Image.open(dataset.items[index][0]).convert("L").resize(
                (dataset.size[1], dataset.size[0]), Image.Resampling.BILINEAR
            )
            raw_label = np.asarray(Image.open(dataset.items[index][1]).convert("L").resize(
                (dataset.size[1], dataset.size[0]), Image.Resampling.NEAREST
            ), dtype=np.uint8)
            ir_rgb = image.convert("RGB")
            label_rgb = colorize_label(raw_label)
            target_rgb = Image.fromarray((target_tensor[0].numpy() * 255).astype(np.uint8)).convert("RGB")
            pred_rgb = Image.fromarray((pred * 255).astype(np.uint8)).convert("RGB")
            overlay = np.asarray(ir_rgb).copy()
            overlay[pred] = [255, 40, 40]
            overlay_rgb = Image.fromarray(overlay, mode="RGB")
            panels = [ir_rgb, label_rgb, target_rgb, pred_rgb, overlay_rgb]
            canvas = Image.new("RGB", (dataset.size[1] * len(panels), dataset.size[0]))
            for panel_index, panel in enumerate(panels):
                canvas.paste(panel, (panel_index * dataset.size[1], 0))
            canvas.save(output_dir / name)
            Image.fromarray((roi_map * 255).clip(0, 255).astype(np.uint8)).save(
                roi_dir / name
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"D:\twj\thermal\MSRS")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="Training device; auto uses CUDA, then Apple MPS, then CPU")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision on CUDA to reduce memory and improve speed")
    parser.add_argument("--target", choices=["contour", "mask"], default="contour")
    parser.add_argument("--height", type=int, default=480,
                        help="Input height; default keeps native MSRS height")
    parser.add_argument("--width", type=int, default=640,
                        help="Input width; default keeps native MSRS width")
    parser.add_argument("--contour-width", type=int, default=3,
                        help="Approximate training contour width in pixels (1, 3, 5, ...)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Validate every N epochs; the final epoch is always validated")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--roi-alpha", type=float, default=1.0,
                        help="ROI residual enhancement strength; 0 disables enhancement")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0,
                        help="CPU threads for PyTorch; 0 keeps the system default")
    parser.add_argument("--out-dir", default="unet_msrs_runs")
    parser.add_argument("--max-vis", type=int, default=20,
                        help="Number of test examples to save in the contour visualization panel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    if args.eval_every < 1:
        raise ValueError("--eval-every must be >= 1")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available. "
                           "Install a CUDA-enabled PyTorch and NVIDIA driver first.")
    if args.device == "mps" and not mps_available:
        raise RuntimeError("--device mps was requested, but MPS is not available. "
                           "Use Apple Silicon with macOS >= 12.3 and a macOS PyTorch build.")
    if args.device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    else:
        selected_device = args.device
    device = torch.device(selected_device)
    # AMP is enabled only for CUDA. MPS support varies by PyTorch/macOS version,
    # so keeping it off avoids unsupported-operation or numerical issues.
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    elif device.type == "mps":
        print("Device: mps (Apple Metal GPU)")
    else:
        print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")

    image_size = (args.height, args.width)
    train_set = MSRSContourDataset(
        args.data_root, "train", image_size, args.target, args.contour_width
    )
    test_set = MSRSContourDataset(
        args.data_root, "test", image_size, args.target, args.contour_width
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    print(f"Train pairs: {len(train_set)}, test pairs: {len(test_set)}, target: {args.target}")

    model = UNet(args.base_channels, roi_alpha=args.roi_alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # torch.cuda.amp is available across recent PyTorch versions; it is
    # disabled automatically on CPU.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc="Overall training",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        tr = run_epoch(
            model, train_loader, optimizer, device, True,
            desc=f"Epoch {epoch:03d}/{args.epochs} train",
            scaler=scaler, use_amp=use_amp,
        )
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            te = run_epoch(
                model, test_loader, optimizer, device, False,
                desc=f"Epoch {epoch:03d}/{args.epochs} test",
                scaler=None, use_amp=use_amp,
            )
            epoch_bar.set_postfix(
                train_loss=f"{tr[0]:.4f}",
                test_f1=f"{te[3]:.4f}",
            )
            print(f"Epoch {epoch:03d} | train loss {tr[0]:.4f} F1 {tr[3]:.4f} | "
                  f"test loss {te[0]:.4f} P/R/F1 {te[1]:.4f}/{te[2]:.4f}/{te[3]:.4f}",
                  flush=True)
            if te[3] > best_f1:
                best_f1 = te[3]
                torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "best.pt")
        else:
            epoch_bar.set_postfix(train_loss=f"{tr[0]:.4f}", test="skip")
            print(f"Epoch {epoch:03d} | train loss {tr[0]:.4f} F1 {tr[3]:.4f} | test skipped",
                  flush=True)

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    save_predictions(model, test_loader, device, out_dir / "test_predictions")
    save_contour_visualizations(model, test_set, device, out_dir / "contour_visualizations", args.max_vis)
    print(f"Saved best.pt and test predictions to {out_dir.resolve()}")
    print(f"Saved contour panels to {(out_dir / 'contour_visualizations').resolve()}")


if __name__ == "__main__":
    main()
