from pathlib import Path

import numpy as np
from PIL import Image


def entropy(image):
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def mutual_information(a, b):
    joint = np.histogram2d(a.ravel(), b.ravel(), bins=256, range=((0, 255), (0, 255)))[0]
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    expected = pa @ pb
    valid = joint > 0
    return float((joint[valid] * np.log2(joint[valid] / expected[valid])).sum())


def average_gradient(image):
    image = image.astype(np.float32)
    gx = image[:, 2:] - image[:, :-2]
    gy = image[2:, :] - image[:-2, :]
    h = min(gx.shape[0], gy.shape[0])
    w = min(gx.shape[1], gy.shape[1])
    return float(np.sqrt((gx[:h, :w] ** 2 + gy[:h, :w] ** 2) / 2).mean())


def spatial_frequency(image):
    image = image.astype(np.float32)
    rf = np.diff(image, axis=1)
    cf = np.diff(image, axis=0)
    return float(np.sqrt(np.mean(rf ** 2) + np.mean(cf ** 2)))


def load(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


root = Path("dataset/TNO")
main_dir = Path("results_main")
fine_dir = Path("results_finetune")
rows = {"main": [], "finetune": []}
for name in sorted(p.stem for p in main_dir.glob("*.bmp")):
    fused_main = load(main_dir / f"{name}.bmp")
    fused_fine = load(fine_dir / f"{name}.bmp")
    vis = load(next((root / "VIS").glob(f"{name}.*")))
    ir = load(next((root / "IR").glob(f"{name}.*")))
    for key, fused in (("main", fused_main), ("finetune", fused_fine)):
        rows[key].append((
            entropy(fused),
            np.std(fused),
            average_gradient(fused),
            spatial_frequency(fused),
            mutual_information(fused, vis) + mutual_information(fused, ir),
        ))

for key, values in rows.items():
    values = np.asarray(values)
    print(key, "EN SD AG SF MI_sum =", np.mean(values, axis=0))
