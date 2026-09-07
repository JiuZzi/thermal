from pathlib import Path

import numpy as np
from PIL import Image


main_dir = Path("results_main")
finetune_dir = Path("results_finetune")
names = sorted(
    {p.name for p in main_dir.glob("*.bmp")}
    & {p.name for p in finetune_dir.glob("*.bmp")}
)
diffs = []
means = []
stds = []
saturation = []
for name in names:
    main = np.asarray(Image.open(main_dir / name).convert("L"), dtype=np.float32)
    finetune = np.asarray(Image.open(finetune_dir / name).convert("L"), dtype=np.float32)
    diffs.append(np.mean(np.abs(main - finetune)))
    means.append((main.mean(), finetune.mean()))
    stds.append((main.std(), finetune.std()))
    saturation.append(
        (np.mean(main <= 2), np.mean(finetune <= 2),
         np.mean(main >= 253), np.mean(finetune >= 253))
    )

print(f"images={len(names)}")
print(f"mean_abs_difference={np.mean(diffs):.3f}")
print(f"max_abs_difference_per_image={np.max(diffs):.3f}")
print(f"images_with_mean_difference_gt_5={np.mean(np.asarray(diffs) > 5):.3f}")
print(f"mean_brightness_main_finetune={np.mean(means, axis=0)}")
print(f"contrast_std_main_finetune={np.mean(stds, axis=0)}")
print(f"dark_saturation_main_finetune={np.mean(saturation, axis=0)[:2]}")
print(f"bright_saturation_main_finetune={np.mean(saturation, axis=0)[2:]}")
print("largest_changes=")
for value, name in sorted(zip(diffs, names), reverse=True)[:8]:
    print(f"  {name}: {value:.3f}")
