from pathlib import Path

import numpy as np
from PIL import Image


def contour(label: np.ndarray, width: int) -> np.ndarray:
    foreground = label > 0
    eroded = foreground.copy()
    eroded[1:, :] &= foreground[:-1, :]
    eroded[:-1, :] &= foreground[1:, :]
    eroded[:, 1:] &= foreground[:, :-1]
    eroded[:, :-1] &= foreground[:, 1:]
    result = foreground & ~eroded
    for _ in range((width - 1) // 2):
        padded = np.pad(result, 1, mode="constant")
        result = (
            padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:]
            | padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:]
            | padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
        )
    return result


labels = Path("MSRS/test/Segmentation_labels")
for directory in (
    Path("ablation_unet_same/test_predictions"),
    Path("ablation_roi_same/test_predictions"),
):
    names = sorted(path.name for path in directory.glob("*.png") if (labels / path.name).exists())
    tp = fp = fn = predicted_pixels = target_pixels = 0
    for name in names:
        prediction = np.asarray(Image.open(directory / name).convert("L")) > 127
        label = np.asarray(Image.open(labels / name).convert("L"))
        target = contour(label, width=3)
        tp += np.logical_and(prediction, target).sum()
        fp += np.logical_and(prediction, ~target).sum()
        fn += np.logical_and(~prediction, target).sum()
        predicted_pixels += prediction.sum()
        target_pixels += target.sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    print(
        f"{directory.parent.name}: images={len(names)}, P={precision:.4f}, "
        f"R={recall:.4f}, F1={f1:.4f}, predicted pixels/image={predicted_pixels / len(names):.0f}, "
        f"target pixels/image={target_pixels / len(names):.0f}"
    )
