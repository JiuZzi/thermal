import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RescaleT, SalObjDataset, ToTensorLab
from model import UIUNET


# UIU-Net paper reproduction on SIRST:
#   train: Misc_1 ... Misc_407
#   test:  Misc_408 ... Misc_427 (handled by test.py)
#
# Expected training layout:
# train_data/
# `-- SIRST407/
#     |-- images/Misc_1.png ... Misc_407.png
#     `-- masks/Misc_1.png  ... Misc_407.png


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    """Apply BCE supervision to the fused output and six side outputs."""
    bce_loss = nn.BCELoss(reduction="mean")

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    print(
        "l0: %3f, l1: %3f, l2: %3f, l3: %3f, "
        "l4: %3f, l5: %3f, l6: %3f"
        % (
            loss0.item(),
            loss1.item(),
            loss2.item(),
            loss3.item(),
            loss4.item(),
            loss5.item(),
            loss6.item(),
        )
    )

    return loss0, loss


def main():
    # Paper/released-code training settings.
    model_name = "uiunet"
    epoch_num = 500
    batch_size_train = 3
    save_frq = 2000

    project_dir = os.getcwd()
    image_dir = os.path.join(project_dir, "train_data", "SIRST407", "images")
    label_dir = os.path.join(project_dir, "train_data", "SIRST407", "masks")
    model_dir = os.path.join(project_dir, "saved_models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Images and copied/renamed masks must share the same stem, e.g.
    # images/Misc_1.png <-> masks/Misc_1.png.
    tra_img_name_list = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    tra_lbl_name_list = [
        os.path.join(label_dir, os.path.basename(image_path))
        for image_path in tra_img_name_list
    ]

    missing_labels = [path for path in tra_lbl_name_list if not os.path.isfile(path)]
    if missing_labels:
        preview = "\n".join(missing_labels[:10])
        raise FileNotFoundError(
            "Training masks are missing. Images and masks must have identical "
            f"filenames. First missing paths:\n{preview}"
        )

    train_num = len(tra_img_name_list)
    if train_num != 407:
        raise RuntimeError(
            "SIRST paper reproduction requires exactly 407 training pairs "
            f"(Misc_1 to Misc_407), but found {train_num}. "
            f"Checked directory: {image_dir}"
        )

    print("---")
    print("train images:", len(tra_img_name_list))
    print("train labels:", len(tra_lbl_name_list))
    print("image directory:", image_dir)
    print("label directory:", label_dir)
    print("model directory:", model_dir)
    print("---")

    # The paper reports 320 x 320, 3-band input. A one-channel infrared image
    # is replicated to three channels by ToTensorLab(flag=0).
    # RandomCrop(288) from the public training script is intentionally omitted
    # because it changes the actual network input from the paper's 320 x 320.
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose(
            [
                RescaleT(320),
                ToTensorLab(flag=0),
            ]
        ),
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )

    # UIU-Net is trained from scratch, as stated in the paper.
    net = UIUNET(3, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    print("device:", device)
    print("---define optimizer...")
    optimizer = optim.Adam(
        net.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    for epoch in range(epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            ite_num4val += 1

            inputs = data["image"].type(torch.FloatTensor)
            labels = data["label"].type(torch.FloatTensor)

            inputs_v = Variable(inputs.to(device), requires_grad=False)
            labels_v = Variable(labels.to(device), requires_grad=False)

            optimizer.zero_grad()

            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss0, loss = muti_bce_loss_fusion(
                d0, d1, d2, d3, d4, d5, d6, labels_v
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_tar_loss += loss0.item()

            del d0, d1, d2, d3, d4, d5, d6, loss0, loss

            print(
                "[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] "
                "train loss: %3f, tar: %3f"
                % (
                    epoch + 1,
                    epoch_num,
                    (i + 1) * batch_size_train,
                    train_num,
                    ite_num,
                    running_loss / ite_num4val,
                    running_tar_loss / ite_num4val,
                )
            )

            if ite_num % save_frq == 0:
                checkpoint_path = os.path.join(
                    model_dir,
                    "%s_bce_itr_%d_train_%3f_tar_%3f.pth"
                    % (
                        model_name,
                        ite_num,
                        running_loss / ite_num4val,
                        running_tar_loss / ite_num4val,
                    ),
                )
                torch.save(net.state_dict(), checkpoint_path)
                print("saved checkpoint:", checkpoint_path)

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

    # test.py expects this exact path/name.
    final_model_path = os.path.join(model_dir, model_name + ".pth")
    torch.save(net.state_dict(), final_model_path)
    print("training finished")
    print("final model saved to:", final_model_path)


if __name__ == "__main__":
    main()
