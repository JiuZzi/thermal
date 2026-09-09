import glob
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RescaleT, SalObjDataset, ToTensorLab
from model import UIUNET
from model.metrics import SamplewiseSigmoidMetric, SigmoidMetric


# UIU-Net paper reproduction on the 20-image SIRST test set:
#   images: Misc_408.jpg ... Misc_427.jpg
#   labels: Misc_408_pixels0.png ... Misc_427_pixels0.png


def norm_pred(prediction):
    """Match the min-max normalization used by the released UIU-Net test code."""
    maximum = torch.max(prediction)
    minimum = torch.min(prediction)
    denominator = maximum - minimum
    if denominator.item() <= 1e-12:
        return torch.zeros_like(prediction)
    return (prediction - minimum) / denominator


def build_test_pairs(image_dir, label_dir):
    """Build deterministic image/label pairs instead of relying on glob order."""
    image_paths = sorted(
        path
        for path in glob.glob(os.path.join(image_dir, "*"))
        if os.path.isfile(path)
        and os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    label_paths = []
    missing_labels = []
    for image_path in image_paths:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, image_id + "_pixels0.png")
        if not os.path.isfile(label_path):
            # Also support already-renamed masks such as Misc_408.png.
            label_path = os.path.join(label_dir, image_id + ".png")
        if not os.path.isfile(label_path):
            missing_labels.append(image_id)
        label_paths.append(label_path)

    if missing_labels:
        raise FileNotFoundError(
            "Missing test labels for: " + ", ".join(missing_labels[:10])
        )

    if len(image_paths) != 20:
        raise RuntimeError(
            "The UIU-Net SIRST paper experiment requires exactly 20 test images "
            f"(Misc_408 to Misc_427), but found {len(image_paths)} in {image_dir}"
        )

    expected_ids = {f"Misc_{index}" for index in range(408, 428)}
    actual_ids = {
        os.path.splitext(os.path.basename(path))[0] for path in image_paths
    }
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise RuntimeError(
            "The SIRST test IDs do not match Misc_408 to Misc_427. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

    return image_paths, label_paths


def main():
    model_name = "uiunet"
    project_dir = os.getcwd()

    image_dir = os.path.join(
        project_dir, "test_data", "Quantification_Results", "test_images"
    )
    label_dir = os.path.join(
        project_dir, "test_data", "Quantification_Results", "test_labels"
    )
    model_path = os.path.join(
        project_dir, "saved_models", model_name, model_name + ".pth"
    )
    result_dir = os.path.join(project_dir, "test_data", model_name + "_results")
    os.makedirs(result_dir, exist_ok=True)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weight not found: {model_path}\n"
            "Run python train.py first, or copy the trained weight to this path."
        )

    image_paths, label_paths = build_test_pairs(image_dir, label_dir)

    print("---")
    print("test images:", len(image_paths))
    print("test labels:", len(label_paths))
    print("model:", model_path)
    print("---")

    test_dataset = SalObjDataset(
        img_name_list=image_paths,
        lbl_name_list=label_paths,
        transform=transforms.Compose(
            [
                RescaleT(320),
                ToTensorLab(flag=0),
            ]
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UIUNET(3, 1)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    # Keep the metric implementations and thresholds from the released code.
    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.55)
    iou_metric.reset()
    niou_metric.reset()

    with torch.no_grad():
        for index, sample in enumerate(test_loader):
            image_name = os.path.basename(image_paths[index])
            print(f"[{index + 1:02d}/20] inferencing: {image_name}")

            inputs = sample["image"].float().to(device)
            labels = sample["label"].float().cpu()

            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            prediction = norm_pred(d0[:, 0, :, :])
            output = prediction.unsqueeze(1).cpu()

            iou_metric.update(output, labels)
            niou_metric.update(output, labels)

            del d0, d1, d2, d3, d4, d5, d6

    pixel_accuracy, iou = iou_metric.get()
    _, niou = niou_metric.get()

    iou = float(iou)
    niou = float(niou)
    pixel_accuracy = float(pixel_accuracy)

    print("--- SIRST paper test result ---")
    print(f"Pixel accuracy: {pixel_accuracy:.6f}")
    print(f"IoU:            {iou:.6f}")
    print(f"nIoU:           {niou:.6f}")
    print("Paper reference: IoU=0.7825, nIoU=0.7515")

    metrics_path = os.path.join(result_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as stream:
        stream.write("UIU-Net SIRST test: Misc_408 to Misc_427\n")
        stream.write(f"Pixel accuracy: {pixel_accuracy:.6f}\n")
        stream.write(f"IoU: {iou:.6f}\n")
        stream.write(f"nIoU: {niou:.6f}\n")
        stream.write("Paper reference: IoU=0.7825, nIoU=0.7515\n")

    print("metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()
