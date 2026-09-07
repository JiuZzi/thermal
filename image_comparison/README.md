# CoCoNet / ISNet image comparison

`generate_comparisons.py` uses the trained weights already present under
`thermal` and creates two kinds of qualitative result:

- MSRS: visible image, infrared image, and CoCoNet fused result.
- IRSTD-1k: infrared image, ground truth, ISNet probability, binary mask,
  and mask overlay.

CoCoNet is an image-fusion model while ISNet is a target-segmentation model,
so their generated pixels do not have the same meaning. The panels keep these
tasks separate and suitable for qualitative inspection.

Run from PowerShell:

```powershell
cd "E:\上海大学\项目\thermal"
conda activate isnet
python .\image_comparison\generate_comparisons.py --num 4
```

Results are written to:

```text
image_comparison\outputs\comparison_overview.png
image_comparison\outputs\msrs\
image_comparison\outputs\irstd_1k\
```

Select explicit samples if needed:

```powershell
python .\image_comparison\generate_comparisons.py `
  --msrs-names 00004N 00055D `
  --irstd-names XDU0 XDU100
```

Use different weights:

```powershell
python .\image_comparison\generate_comparisons.py `
  --coconet-weight ".\CoCoNet-main\logs_main_gpu\latest.pth" `
  --isnet-weight ".\ISNet-master\result\RUN\checkpoint\MODEL.pkl"
```

The source repositories and datasets are only read; the script writes solely
inside its output directory unless `--output-dir` is supplied.
