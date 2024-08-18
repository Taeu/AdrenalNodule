import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image
import pandas as pd
import pickle
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from skimage import io, exposure
import cv2
import matplotlib.pyplot as plt
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from albumentations import (Flip, OneOf, Compose, Rotate)


def album_aug(p=0.5):
    return Compose([
        Rotate(),
        Flip(),], p=p,
    additional_targets={"image2" : "image", "image3" : "image"})

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def get_transform(phase = 'train', method=Image.BILINEAR):
    transform_list = []

    if phase == 'train':
        transform_list.append(transforms.ToPILImage(mode= "RGB"))
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomVerticalFlip())
        transform_list.append(transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))) ###
    # 더 추가하고,
    return transforms.Compose(transform_list)

def debug_fn(a, b, c):
    print(a, b, c)
    import pdb; pdb.set_trace()


class CustomDataset(data.Dataset):
    def __init__(self, args, phase='train', transform_train=None, transform_val = None, img_channel = 1):
        self.args = args
        self.root = args.root
        self.phase = phase
        self.img_channel = img_channel
        #self.labels = {}
        self.data_path = os.path.join(self.root)
        # AGN_only, AGN_slice, AGN_with_AG
        if phase == 'train' or phase == 'valid' or self.args.test_for_train :
            self.image_dir = os.path.join(self.root, self.args.image_dir)
            self.label_dir = os.path.join(self.root, self.args.label_dir)
            self.output_dir = os.path.join(self.root, self.args.output_dir)
        elif phase == 'test':
            self.image_dir = os.path.join(self.root, self.args.image_dir.replace('train','test'))
            self.label_dir = os.path.join(self.root, self.args.label_dir.replace('train','test'))
            self.output_dir = os.path.join(self.root, self.args.output_dir.replace('train','test'))

        self.image_list = sorted(os.listdir(self.output_dir))

        s, e = 0, len(self.image_list)
        if self.phase == 'train':
            e = -len(self.image_list) // 8
        elif self.phase == 'valid':
            s = -len(self.image_list) // 8
        self.image_list = self.image_list[s:e]
        self.transform_train = transform_train #get_transform()
        self.transform_val = transform_val

    #@torch.jit.script_method
    def __getitem__(self, index):
        # train index
        fn = self.image_list[index]
        image = np.load(os.path.join(self.image_dir, fn))
        label = np.load(os.path.join(self.label_dir, fn))
        output = np.load(os.path.join(self.output_dir, fn))
        label2 = np.array(np.sum(label) != 0).astype('float32')
        if self.args.image_size != 160:
            image = cv2.resize(image, dsize=(self.args.image_size, self.args.image_size))
            label = cv2.resize(label, dsize=(self.args.image_size, self.args.image_size))
            output = cv2.resize(output, dsize=(self.args.image_size, self.args.image_size))

        if self.img_channel == 1:
            image_output = image * output
            input = np.zeros((image.shape[0], image.shape[1],3))
            if 'input' in self.args.exp_name:
                input[:,:,0] = image
            else:
                input[:,:,0] = image_output
                label = label * output
            if 'with_all_input' in self.args.exp_name:
                input[:,:,1] = image
                input[:,:,2] = image
            else:
                input[:, :, 1] = image_output
                input[:, :, 2] = image_output
            input = self.windowing(input)
            new_img = input.astype(np.uint8)

            if self.phase == 'train':
                new_label = np.expand_dims(label, axis=-1).astype(np.int8)
                new_label = SegmentationMapsOnImage(new_label, shape=new_img.shape)
                new_img, label = self.transform_train(image = new_img, segmentation_maps = new_label)
                label = label.get_arr()

            new_img = self.transform_val(new_img)

        elif self.img_channel == 3:
            # 3 consecutive slice
            if self.phase == 'train':
                aug_input = {'image': image, 'image2': output, 'image3': label}
                aug_output = self.transform_train(**aug_input)
                image, output, label = aug_output['image'], aug_output['image2'], aug_output['image3']

            label2 = label.max()
            label = (label >= 3) * 1
            label = label.transpose((2,0,1))

            #import pdb; pdb.set_trace()
            new_img = self.transform_val(image.astype(np.uint8))
            new_output = self.transform_val(output.astype(np.uint8))
            new_img = torch.cat((new_img, new_output), 0)

        label = torch.tensor(label)
        label2 = torch.tensor(label2)

        if self.phase == 'train':
            label = label.squeeze()
            # new_img = new_img.unsqueeze()
            #print(new_img.shape, label.shape)
        if self.phase == 'test':
            return new_img, label, label2, fn
        # -------------------------------
        return new_img, label, label2
        # -------------------------------

    def __len__(self):
        return len(self.image_list)

    def get_label_file(self):
        return self.label_path

    def windowing(self, im, win = [-350, 350]):
        """scale intensity from win[0]~win[1] to float numbers in 0~255"""
        im1 = im.astype(float)
        im1 -= win[0]
        im1 /= win[1] - win[0]
        im1[im1 > 1] = 1
        im1[im1 < 0] = 0
        im1 *= 255
        return im1

def data_loader(args, phase='train', batch_size=16):

    tfs1 = iaa.Sequential([
            iaa.flip.Fliplr(p=0.5),
            # iaa.flip.Flipud(p=0.5),
            iaa.Rotate((-15, 15)),
            # iaa.GaussianBlur(sigma=(0.0, 0.1)),
            iaa.MultiplyBrightness(mul=(1.0, 1.0)),
            # iaa.size.Resize((args.image_size, args.image_size))
        ])


    tfs_val = transforms.Compose([
        # iaa.Sequential([
        #     iaa.MultiplyBrightness(mul=(1.0, 1.0)),
        #     iaa.size.Resize((args.image_size, args.image_size))
        # ]).augment_image,
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])

    if phase == 'train':
        shuffle = True
        transform_train = tfs1
    else:
        shuffle = False
        transform_train = None
    transform_val = tfs_val

    dataset = CustomDataset(args, phase, transform_train, transform_val)
    if phase == 'test':
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    else:
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16)
    return dataloader

def data_loader3(args, phase='train', batch_size = 16):
    tfs1 =album_aug(p=0.5)

    tfs_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])

    if phase == 'train':
        shuffle = True
        transform_train = tfs1
    else:
        shuffle = False
        transform_train = None
    transform_val = tfs_val

    dataset = CustomDataset(args, phase, transform_train, transform_val, img_channel=3)
    if phase == 'test':
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    else:
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16)
    return dataloader