import os
import random
import torch
import cv2
import argparse
import numpy as np
import imageio
from PIL import Image
from torchvision import transforms

from models.model import Unet_resize_conv
from utils import fname_presuffix
from models.train_tasks import train, finetune_multiNegative


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # change the CUDA index in your need

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
argparser.add_argument('--num_task', type=int, help='k shot for support set', default=3)
argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=1e-4)
argparser.add_argument('--bs', type=int, help='batch size', default=10)
argparser.add_argument('--logdir', type=str, default='./logs/')
argparser.add_argument('--train', action='store_true')
argparser.add_argument('--test', action='store_true')
argparser.add_argument('--test_vis', type=str, help='Directory of the test visible images')
argparser.add_argument('--test_ir', type=str, help='Directory of the test infrared images')
argparser.add_argument('--resume', action='store_true')
argparser.add_argument('--resume_ckpt', type=str, default='logs/')
argparser.add_argument('--ckpt', type=str, default='./logs/latest.pth')
argparser.add_argument('--finetune', action='store_true')
argparser.add_argument('--use_gpu', action='store_true')
argparser.add_argument('--w', action='store_true')
argparser.add_argument('--fs', type=int, help='fusion strategy,0~6', default=0)
argparser.add_argument('--task', type=int, help='task 0,1,2(visir,me,mf)', default=0)
argparser.add_argument('--save_dir', type=str, default='./results/')
argparser.add_argument('--c1', type=float, help='weight grad', default=0.5)
argparser.add_argument('--c2', type=float, help='weight entropy', default=0.5)
argparser.add_argument('--contrast', type=float, help='contrastive loss weight', default=1.0)
argparser.add_argument('--w_loss', type=float, help='weight of self-adaptive loss', default=1.0)
argparser.add_argument(
    '--ablation', type=str, choices=['full', 'no_vgg', 'no_ca', 'no_ca_vgg'],
    default='full',
    help='Architecture ablation: full; no_vgg; no_ca; or no_ca_vgg'
)
argparser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible ablations')

args = argparser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GrayscaleTransform:
    def __call__(self, img):
        # Convert the image to grayscale
        if img.shape[0] == 3:
            img = img[0, :, :]
            img = torch.unsqueeze(img, 0)
        return img


def test(args, model, vis_path, ir_path, save_path, prefix='', suffix='', ext='.bmp'):

    checkpath = args.ckpt
    print('Loading from {}...'.format(checkpath))

    vis_list = [n for n in os.listdir(vis_path)]
    ir_list = vis_list

    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda')
    logs = torch.load(checkpath, map_location=device)  # use checkpoints when testing
    model.load_state_dict(logs['state_dict'])
    model.to(device)    

    transform = transforms.Compose([
        transforms.ToTensor(),
        GrayscaleTransform()
    ])

    import time
    Time = []
    for vis_, ir_ in zip(vis_list, ir_list):
        fn_ir = os.path.join(ir_path, ir_)
        fn_vis = os.path.join(vis_path, vis_)
        start = time.time()

        img_vis = imageio.imread(fn_vis).astype(np.float32)
        img_ir = imageio.imread(fn_ir).astype(np.float32)

        # to tensor and grayscale
        data_vis = transform(img_vis)
        data_ir = transform(img_ir)

        # add batch size dimension
        data_vis = torch.unsqueeze(data_vis, 0).to(device)
        data_ir = torch.unsqueeze(data_ir, 0).to(device)
        output = model(data_vis, data_ir)

        output = np.transpose(
            (torch.squeeze(output, 0).cpu().detach().numpy() * 127.5 + 127.5),
            axes=(1, 2, 0)
        )
        # OpenCV does not reliably write float32 BMP images. Convert the
        # normalized network output to a valid 8-bit image before saving.
        output = np.clip(output, 0, 255).astype(np.uint8)
        # The fusion network returns one grayscale channel. OpenCV expects a
        # 2-D array for grayscale BMP output rather than (H, W, 1).
        if output.ndim == 3 and output.shape[2] == 1:
            output = output[:, :, 0]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_fn = fname_presuffix(
            fname=vis_, prefix=prefix,
            suffix=suffix, newpath=save_path)
        output_path = save_fn.split('.')[0] + ext
        # PIL handles Unicode Windows paths (including the Chinese project
        # directory) reliably, whereas cv2.imwrite may silently fail there.
        Image.fromarray(output, mode='L').save(output_path)

        end = time.time()
        Time.append(end - start)

    print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))


def main():

    set_seed(args.seed)

    print(
        '\nCoCoNet: Coupled Contrastive Learning Network with Multi-level Feature Ensemble for Multi-modality Image Fusion\n')
    print('Cuda ', torch.cuda.is_available())
    print('Training', args.train)
    print('Ablation', args.ablation)
    print('Seed', args.seed)

    if args.use_gpu:
        model = Unet_resize_conv(ablation=args.ablation).to(torch.device('cuda'))
    else:
        model = Unet_resize_conv(ablation=args.ablation).to(torch.device('cpu'))

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    print('Total trainable tensors:', num)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.train:
        model.train()
        data_path = './training.h5'
        train(model, data_path, optim, args)

    elif args.test:
        # TEST DIRECTORY
        dir_vis = args.test_vis
        dir_ir = args.test_ir
        save_path = args.save_dir
        test(args, model, dir_vis, dir_ir, save_path, suffix='')
    
    elif args.finetune:
        model.train()
        ft_data_path = './training_mask.h5'
        finetune_multiNegative(model, ft_data_path, optim, args)


if __name__ == '__main__':
    main()
