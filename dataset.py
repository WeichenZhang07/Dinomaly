import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop((isize, isize), scale=(0.6, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class LOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*/000.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        size = (img.size[1], img.size[0])
        img = self.transform(img)
        type = self.types[idx]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, type, size


class InsPLADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)

        return img, label, img_path


class AeBADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        defect_types = [i for i in defect_types if i[0] != '.']
        for defect_type in defect_types:
            if defect_type == 'good':
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend([0] * len(img_paths))
                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))
            else:
                domain_types = os.listdir(os.path.join(self.img_path, defect_type))
                domain_types = [i for i in domain_types if i[0] != '.']

                for domain_type in domain_types:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, domain_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, domain_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)
                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return img, label
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):

        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
            img_tot_paths.extend(img_paths)
            tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path, label = self.img_paths[idx], self.labels[idx]
            img = Image.open(img_path).convert('RGB')
        except:
            img_path, label = self.img_paths[idx - 1], self.labels[idx - 1]
            img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label


class MVTecDRAEMDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, strong_transform, phase, anomaly_source_path, anomaly_ratio=0.5,
                 size=256):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.strong_transform = strong_transform
        self.anomaly_ratio = anomaly_ratio
        self.size = size
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        no_anomaly = random.random()
        if no_anomaly > self.anomaly_ratio:
            return image, 0
        else:
            aug = self.randAugmenter()

            perlin_scale = 6
            min_perlin_scale = 0
            anomaly_source_img = Image.open(anomaly_source_path).convert('RGB').resize((self.size, self.size))
            anomaly_source_img = np.asarray(anomaly_source_img)
            anomaly_img_augmented = aug(image=anomaly_source_img)

            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((self.size, self.size),
                                             (perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            threshold = 0.5
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)

            img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr

            beta = random.random() * 0.7 + 0.1

            image = image.resize((self.size, self.size))
            image = np.asarray(image)
            augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
            # augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image

            return Image.fromarray(np.uint8(augmented_image)), 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        a_img, label = self.augment_image(img, self.anomaly_source_paths[anomaly_source_idx])

        img = self.transform(img)
        a_img = self.strong_transform(a_img)

        assert img.size()[1:] == a_img.size()[1:], "image.size != a_img.size !!!"

        return img, a_img, label


class MVTecSimplexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform

        self.simplexNoise = Simplex_CLASS()
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img_normal = self.transform(img)

        if random.random() > 0.5:
            return img_normal, img_normal
        ## simplex_noise
        size = 256
        img = img.resize((size, size))
        img = np.asarray(img)
        h_noise = np.random.randint(10, int(size // 8))
        w_noise = np.random.randint(10, int(size // 8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256, 256, 3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise + w_noise,
        :] = 0.2 * simplex_noise.transpose(1, 2, 0)
        img_noise = img + init_zero * 255
        img_noise = Image.fromarray(np.uint8(img_noise))
        img_noise = self.transform(img_noise)

        return img_normal, img_noise
import os
from enum import Enum
import warnings

import PIL
from PIL import Image
import torch
from torchvision import transforms


class AspectRatioResize:
    """
    保持宽高比：将短边缩放到 target_size。
    可能导致长边 > target_size，后续通常用 CenterCrop 截取。
    """
    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        self.size = size
        self.interp = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < h:
            new_w = self.size
            new_h = int(round(h * self.size / w))
        else:
            new_h = self.size
            new_w = int(round(w * self.size / h))
        return img.resize((new_w, new_h), self.interp)


class LongestSideResize:
    """
    将长边缩放到 target_size，保持宽高比。适合随后 Pad 到方形不裁剪内容。
    """
    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        self.size = size
        self.interp = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        if longest == self.size:
            return img
        scale = self.size / longest
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), self.interp)


class PadToSquare:
    """Pad 图像到 (target_size, target_size)。支持 constant | reflect | replicate。"""
    def __init__(self, size, mode='reflect', value=0.0):
        self.size = size
        self.mode = mode
        self.value = value

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == self.size and h == self.size:
            return img
        # 若超过目标尺寸（理论上不会，如果前面缩放长边=target）则裁剪中心
        if w > self.size or h > self.size:
            left = max(0, (w - self.size) // 2)
            top = max(0, (h - self.size) // 2)
            img = img.crop((left, top, left + min(self.size, w), top + min(self.size, h)))
            w, h = img.size
        pad_l = (self.size - w) // 2
        pad_r = self.size - w - pad_l
        pad_t = (self.size - h) // 2
        pad_b = self.size - h - pad_t
        if self.mode == 'constant':
            color = tuple(int(self.value * 255) for _ in range(3))
            canvas = Image.new('RGB', (self.size, self.size), color)
            canvas.paste(img, (pad_l, pad_t))
            return canvas
        # 对 reflect / replicate 使用 tensor pad 再转回 PIL
        import torchvision.transforms.functional as TF
        tensor = TF.to_tensor(img).unsqueeze(0)  # (1,C,H,W)
        pad = [pad_l, pad_t, pad_r, pad_b]  # left, top, right, bottom
        mode = 'reflect' if self.mode == 'reflect' else 'replicate'
        padded = torch.nn.functional.pad(tensor, (pad[0], pad[2], pad[1], pad[3]), mode=mode)
        padded = padded.clamp(0, 1)
        return TF.to_pil_image(padded.squeeze(0))


# MVTec AD 2.0 官方类别
MVTEC_AD2_CLASSNAMES_LIST = {
    'can': 0,
    'fabric': 1,
    'fruit_jelly': 2,
    'rice': 3,
    'sheet_metal': 4,
    'vial': 5,
    'wallplugs': 6,
    'walnuts': 7,
}
MVTEC_AD2_CLASSNAMES = list(MVTEC_AD2_CLASSNAMES_LIST.keys())
# 返回一个封装成Tensor的类别ID[B,1]
def classname_to_idTensor(classname):
    class_id =  MVTEC_AD2_CLASSNAMES_LIST.get(classname, -1)
    #class_id转换成Tensor
    return torch.tensor([[class_id]], dtype=torch.long)
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

import os
import PIL.Image
import torch
from torchvision import transforms

# your existing constants
MVTEC_AD2_CLASSNAMES = list(MVTEC_AD2_CLASSNAMES_LIST.keys())

class MVTecAD2Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        preserve_aspect_ratio=False,
        pad_to_square=False,
        pad_mode='reflect',
        pad_value=0.0,
        resize_strategy='short_side',
        center_crop=True,
        tile_size=None,        # ---- NEW -----
        tile_stride=None,      # ---- NEW -----
        global_size=224,       # ---- NEW -----
        **kwargs,
    ):
        super().__init__()

        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else MVTEC_AD2_CLASSNAMES
        self.train_val_split = train_val_split
        self.preserve_aspect_ratio = preserve_aspect_ratio

        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD

        # TILE mode parameters
        self.tile_size = tile_size
        self.tile_stride = tile_stride or tile_size
        self.global_size = global_size
        self.tile_index = None
        self.global_cache = {}

        # load paths
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        # ------------------------ Transforms ------------------------
        img_tfms = []
        mask_tfms = []

        if resize_strategy == 'longest':
            img_tfms.append(LongestSideResize(imagesize))
            mask_tfms.append(LongestSideResize(imagesize, interpolation=PIL.Image.NEAREST))
            if pad_to_square:
                img_tfms.append(PadToSquare(imagesize, mode=pad_mode, value=pad_value))
                mask_tfms.append(PadToSquare(imagesize, mode=pad_mode, value=0))
        elif resize_strategy == 'short_side':
            if preserve_aspect_ratio:
                img_tfms.append(AspectRatioResize(imagesize))
                mask_tfms.append(AspectRatioResize(imagesize, interpolation=PIL.Image.NEAREST))
                if center_crop:
                    img_tfms.append(transforms.CenterCrop(imagesize))
                    mask_tfms.append(transforms.CenterCrop(imagesize))
            else:
                img_tfms.append(transforms.Resize((imagesize, imagesize)))
                mask_tfms.append(transforms.Resize((imagesize, imagesize), interpolation=PIL.Image.NEAREST))
        elif resize_strategy == 'none' and pad_to_square:
            img_tfms.append(PadToSquare(imagesize, mode=pad_mode, value=pad_value))
            mask_tfms.append(PadToSquare(imagesize, mode=pad_mode, value=0))

        img_tfms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.transform_mean, std=self.transform_std)
        ])
        mask_tfms.append(transforms.ToTensor())

        self.transform_img = transforms.Compose(img_tfms)
        self.transform_mask = transforms.Compose(mask_tfms)

        if self.tile_size is not None:
            self._build_tile_index()


    # ------------------------------------------------------------------
    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        split_name = self.split.value

        if split_name == 'train':
            possible_dirs = ['train']
        elif split_name == 'val':
            possible_dirs = ['validation', 'val']
        elif split_name == 'test':
            possible_dirs = ['test_public']
        else:
            possible_dirs = [split_name]

        for classname in self.classnames_to_use:
            if not isinstance(classname, str):
                continue

            selected_dir = None
            for cand in possible_dirs:
                cand_dir = os.path.join(self.source, classname, cand)
                if os.path.isdir(cand_dir):
                    selected_dir = cand_dir
                    break
            if selected_dir is None:
                selected_dir = os.path.join(self.source, classname, possible_dirs[0])

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in ["good", "bad"]:
                anomaly_dir = os.path.join(selected_dir, anomaly)
                if os.path.isdir(anomaly_dir):
                    files = sorted(os.listdir(anomaly_dir))
                    imgpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_dir, f) for f in files if os.path.isfile(os.path.join(anomaly_dir, f))
                    ]
                    if split_name == "test" and anomaly == "bad":
                        mask_dir = os.path.join(selected_dir, "ground_truth", "bad")
                        masks = sorted(os.listdir(mask_dir)) if os.path.isdir(mask_dir) else []
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(mask_dir, m) for m in masks
                        ]
                    else:
                        maskpaths_per_class[classname][anomaly] = [None] * len(imgpaths_per_class[classname][anomaly])

        data_list = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, imgp in enumerate(imgpaths_per_class[classname][anomaly]):
                    maskp = maskpaths_per_class[classname][anomaly][i]
                    data_list.append([classname, anomaly, imgp, maskp])

        return imgpaths_per_class, data_list


    # ------------------------------------------------------------------
    def _build_tile_index(self):
        self.tile_index = []
        for classname, anomaly, imgp, maskp in self.data_to_iterate:
            img = PIL.Image.open(imgp).convert("RGB")
            W, H = img.size

            cols = max(1, (W - self.tile_size) // self.tile_stride + 1)
            rows = max(1, (H - self.tile_size) // self.tile_stride + 1)

            for r in range(rows):
                for c in range(cols):
                    x0 = c * self.tile_stride
                    y0 = r * self.tile_stride
                    x1 = x0 + self.tile_size
                    y1 = y0 + self.tile_size

                    self.tile_index.append({
                        "classname": classname,
                        "anomaly": anomaly,
                        "image_path": imgp,
                        "mask_path": maskp,
                        "raw_size": (H, W),
                        "tile_box": (x0, y0, x1, y1),
                    })


    # ------------------------------------------------------------------
    def _getitem_tile(self, idx):
        info = self.tile_index[idx]

        img = PIL.Image.open(info["image_path"]).convert("RGB")
        W, H = info["raw_size"]
        x0, y0, x1, y1 = info["tile_box"]

        # Tile crop + resize
        tile = img.crop((x0, y0, x1, y1))
        tile = tile.resize((self.tile_size, self.tile_size), PIL.Image.BILINEAR)
        tile = transforms.ToTensor()(tile)
        tile = transforms.Normalize(self.transform_mean, self.transform_std)(tile)

        # Mask tile
        if self.split == DatasetSplit.TEST and info["mask_path"] is not None:
            mask_full = PIL.Image.open(info["mask_path"])
            mask = transforms.ToTensor()(mask_full.crop((x0, y0, x1, y1)))
        else:
            mask = torch.zeros([1, self.tile_size, self.tile_size])

        # Cached global image
        if info["image_path"] not in self.global_cache:
            g = PIL.Image.open(info["image_path"]).convert("RGB")
            g = g.resize((self.global_size, self.global_size), PIL.Image.BILINEAR)
            g = transforms.ToTensor()(g)
            g = transforms.Normalize(self.transform_mean, self.transform_std)(g)
            self.global_cache[info["image_path"]] = g

        tile_coord = torch.tensor([x0/W, y0/H, x1/W, y1/H], dtype=torch.float32)

        return {
            "image": tile,
            "mask": mask,
            "classname": info["classname"],
            "classid": classname_to_idTensor(info["classname"]),
            "anomaly": info["anomaly"],
            "is_anomaly": int(info["anomaly"] != "good"),
            "image_path": info["image_path"],
            "tile_coord": tile_coord,
            "global_image": self.global_cache[info["image_path"]],
        }


    # ------------------------------------------------------------------
    def _getitem_normal(self, idx):
        classname, anomaly, imgp, maskp = self.data_to_iterate[idx]
        img = PIL.Image.open(imgp).convert("RGB")
        img = self.transform_img(img)

        if self.split == DatasetSplit.TEST and maskp is not None:
            mask = PIL.Image.open(maskp)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *img.size()[1:]])

        return {
            "image": img,
            "mask": mask,
            "classname": classname,
            "classid": classname_to_idTensor(classname),
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(imgp.replace("\\", "/").split("/")[-4:]),
            "image_path": imgp,
        }


    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        return self._getitem_tile(idx) if self.tile_size else self._getitem_normal(idx)

    def __len__(self):
        return len(self.tile_index) if self.tile_size else len(self.data_to_iterate)
