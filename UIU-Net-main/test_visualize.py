import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RescaleT, SalObjDataset, ToTensorLab
from model import UIUNET


# The released SIRST IoU implementation thresholds the normalized prediction
# at 0.22. Use the same threshold for the binary masks shown here.
PREDICTION_THRESHOLD = 0.22


def imread_unicode(path, flags):
    """Read an image from a Windows path that may contain non-ASCII text."""
    try:
        encoded_data = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise RuntimeError(f"Failed to open image file: {path}") from error

    if encoded_data.size == 0:
        raise RuntimeError(f"Image file is empty: {path}")

    image = cv2.imdecode(encoded_data, flags)
    if image is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return image


def imwrite_unicode(path, image):
    """Write an image to a Windows path that may contain non-ASCII text."""
    extension = os.path.splitext(path)[1]
    if not extension:
        raise RuntimeError(f"Output path has no image extension: {path}")

    success, encoded_data = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"Failed to encode output image: {path}")

    try:
        encoded_data.tofile(path)
    except OSError as error:
        raise RuntimeError(f"Failed to write output image: {path}") from error


def norm_pred(prediction):
    maximum = torch.max(prediction)
    minimum = torch.min(prediction)
    denominator = maximum - minimum
    if denominator.item() <= 1e-12:
        return torch.zeros_like(prediction)
    return (prediction - minimum) / denominator


def build_test_pairs(image_dir, label_dir):
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
            "Expected 20 SIRST test images (Misc_408 to Misc_427), "
            f"but found {len(image_paths)} in {image_dir}"
        )

    return image_paths, label_paths


def add_panel_title(image, title):
    title_height = 34
    canvas = np.zeros(
        (image.shape[0] + title_height, image.shape[1], 3), dtype=np.uint8
    )
    canvas[title_height:, :] = image
    cv2.putText(
        canvas,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_contours(original_bgr, ground_truth, prediction):
    overlay = original_bgr.copy()

    gt_contours, _ = cv2.findContours(
        ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    pred_contours, _ = cv2.findContours(
        prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Green: ground truth. Red: UIU-Net prediction.
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 1)
    cv2.drawContours(overlay, pred_contours, -1, (0, 0, 255), 1)
    cv2.putText(
        overlay,
        "GT: green  Prediction: red",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


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

    output_root = os.path.join(project_dir, "test_data", "uiunet_visualization")
    probability_dir = os.path.join(output_root, "probability")
    prediction_dir = os.path.join(output_root, "prediction")
    overlay_dir = os.path.join(output_root, "overlay")
    comparison_dir = os.path.join(output_root, "comparison")
    for directory in (
        probability_dir,
        prediction_dir,
        overlay_dir,
        comparison_dir,
    ):
        os.makedirs(directory, exist_ok=True)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weight not found: {model_path}\n"
            "Run python train.py first."
        )

    image_paths, label_paths = build_test_pairs(image_dir, label_dir)

    dataset = SalObjDataset(
        img_name_list=image_paths,
        lbl_name_list=label_paths,
        transform=transforms.Compose(
            [
                RescaleT(320),
                ToTensorLab(flag=0),
            ]
        ),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UIUNET(3, 1)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    print("device:", device)
    print("model:", model_path)
    print("output:", output_root)

    with torch.no_grad():
        for index, sample in enumerate(loader):
            image_path = image_paths[index]
            label_path = label_paths[index]
            image_id = os.path.splitext(os.path.basename(image_path))[0]

            inputs = sample["image"].float().to(device)
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)

            prediction_320 = norm_pred(d0[:, 0, :, :])[0].cpu().numpy()

            original_bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
            original_height, original_width = original_bgr.shape[:2]

            probability = cv2.resize(
                prediction_320,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )
            probability_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
            prediction_u8 = (
                probability >= PREDICTION_THRESHOLD
            ).astype(np.uint8) * 255

            ground_truth = imread_unicode(label_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.resize(
                ground_truth,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )
            ground_truth = (ground_truth > 0).astype(np.uint8) * 255

            overlay = draw_contours(original_bgr, ground_truth, prediction_u8)

            gt_bgr = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR)
            prediction_bgr = cv2.cvtColor(prediction_u8, cv2.COLOR_GRAY2BGR)
            comparison = np.concatenate(
                [
                    add_panel_title(original_bgr, "IR IMAGE"),
                    add_panel_title(gt_bgr, "GROUND TRUTH"),
                    add_panel_title(prediction_bgr, "PREDICTION"),
                    add_panel_title(overlay, "CONTOUR OVERLAY"),
                ],
                axis=1,
            )

            imwrite_unicode(
                os.path.join(probability_dir, image_id + "_probability.png"),
                probability_u8,
            )
            imwrite_unicode(
                os.path.join(prediction_dir, image_id + "_prediction.png"),
                prediction_u8,
            )
            imwrite_unicode(
                os.path.join(overlay_dir, image_id + "_overlay.png"), overlay
            )
            imwrite_unicode(
                os.path.join(comparison_dir, image_id + "_comparison.png"),
                comparison,
            )

            print(f"[{index + 1:02d}/20] saved: {image_id}")
            del d0, d1, d2, d3, d4, d5, d6

    print("visualization finished")
    print("results saved to:", output_root)


if __name__ == "__main__":
    main()
