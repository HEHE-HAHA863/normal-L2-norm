#!/usr/bin/env python
# encoding: utf-8

import os
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from PIL import Image
from os import listdir
from os.path import join


def get_args(parser):
    parser.add_argument('--dataset', default= "cifar10", help='mnist | cifar10 | cifar100 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', default="../data", help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='number of channel')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--max_iter', type=int, default=300000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.00005')
    parser.add_argument('--mmd_kernel_multipliers', default='1.0', help='comma-separated multipliers for the median squared-L2 Gaussian kernel denominator, e.g. 1.0 or 1.0,0.5,0.2,0.1')
    parser.add_argument('--gpu_device', type=int, default=0, help='using gpu device id')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    return parser


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class FolderWithImages(data.Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        super(FolderWithImages, self).__init__()
        self.image_filenames = [join(root, x)
                                for x in listdir(root) if is_image_file(x.lower())]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class ALICropAndScale(object):
    def __call__(self, img):
        return img.resize((64, 78), Image.Resampling.LANCZOS).crop((0, 7, 64, 64 + 7))

def get_data(args, train_flag=True):
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if args.dataset in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(
            root=args.dataroot,
            transform=transform
        )

    elif args.dataset == 'lsun':
        dataset = dset.LSUN(
            db_path=args.dataroot,
            classes=['bedroom_train'],
            transform=transform
        )

    elif args.dataset == 'cifar10':
        cifar10_dir = os.path.join(args.dataroot, 'cifar-10-batches-py')
        download_flag = not os.path.exists(cifar10_dir)

        print(f'[cifar10] data dir: {cifar10_dir}')
        print(f'[cifar10] download = {download_flag}')

        dataset = dset.CIFAR10(
            root=args.dataroot,
            download=download_flag,
            train=train_flag,
            transform=transform
        )

    elif args.dataset == 'cifar100':
        cifar100_dir = os.path.join(args.dataroot, 'cifar-100-python')
        download_flag = not os.path.exists(cifar100_dir)

        print(f'[cifar100] data dir: {cifar100_dir}')
        print(f'[cifar100] download = {download_flag}')

        dataset = dset.CIFAR100(
            root=args.dataroot,
            download=download_flag,
            train=train_flag,
            transform=transform
        )

    elif args.dataset == 'mnist':
        mnist_processed_dir = os.path.join(args.dataroot, 'MNIST')
        download_flag = not os.path.exists(mnist_processed_dir)

        print(f'[mnist] data dir: {mnist_processed_dir}')
        print(f'[mnist] download = {download_flag}')

        dataset = dset.MNIST(
            root=args.dataroot,
            download=download_flag,
            train=train_flag,
            transform=transform
        )

    elif args.dataset == 'celeba':
        imdir = 'train' if train_flag else 'val'
        dataroot = os.path.join(args.dataroot, imdir)

        if args.image_size != 64:
            raise ValueError('the image size for CelebA dataset need to be 64!')

        dataset = FolderWithImages(
            root=dataroot,
            input_transform=transforms.Compose([
                ALICropAndScale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            target_transform=transforms.ToTensor()

        )

    else:
        raise ValueError("Unknown dataset %s" % (args.dataset))

    return dataset

def normalize(x, dim=1):
    return x.div(x.norm(2, dim=dim).expand_as(x))


def match(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean()
    elif dist == 'L1':
        return (x - y).abs().mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)
        return 2 - (x_n).mul(y_n).mean()
    else:
        assert dist == 'none', 'wtf ?'
