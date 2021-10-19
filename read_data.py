# -*- coding: utf-8 -*-

import torch.utils.data as data
from config import *
import os
from PIL import Image


class DeepFashionDataset(data.Dataset):
    def __init__(self, data_type="train", transform=None, target_transform=None, crop=False, img_path=None):
        self.transform = transform
        self.target_transform = target_transform
        self.crop = crop

        self.data_type = data_type
        if data_type == "single":
            self.img_path = img_path
            return

        self.train_list = []
        self.val_list = []
        self.test_list = []

        self.attributes = dict()

        if self.data_type == "test":
            self.read_test_images()
        else:
            self.read_attribute_labels()

        self.bbox = dict()
        self.read_bbox()

    def __len__(self):
        if self.data_type == "train":
            return len(self.train_list)
        elif self.data_type == "val":
            return len(self.val_list)
        elif self.data_type == "test":
            return len(self.test_list)
        else:
            return 1

    def read_test_images(self):
        images_file_path = os.path.join(DATASET_BASE, 'split', '{}.txt'.format(self.data_type))
        images_lines = self.read_lines(images_file_path)
        for image in images_lines:
            image = image.rstrip("\n")
            self.test_list.append(image)  # save image path

    def read_attribute_labels(self):
        images_file_path = os.path.join(DATASET_BASE, 'split', '{}.txt'.format(self.data_type))
        category_file_path = os.path.join(DATASET_BASE, 'split', '{}_attr.txt'.format(self.data_type))

        images_lines = self.read_lines(images_file_path)
        category_lines = self.read_lines(category_file_path)
        for image, category in zip(images_lines, category_lines):
            image = image.rstrip("\n")
            if self.data_type == "train":
                self.train_list.append(image)  # save image path
            else:
                self.val_list.append(image)  # save image path
            attribute_list = list(map(int, category.split()))  # read attributes of 6 categories
            self.attributes[image] = attribute_list  # map categories to image

    def read_bbox(self):
        images_file_path = os.path.join(DATASET_BASE, 'split', '{}.txt'.format(self.data_type))
        images_lines = self.read_lines(images_file_path)
        bbox_file_path = os.path.join(DATASET_BASE, r'split', r'{}_bbox.txt'.format(self.data_type))
        bbox_lines = self.read_lines(bbox_file_path)
        for image, line in zip(images_lines, bbox_lines):
            image = image.rstrip("\n")
            self.bbox[image] = list(map(int, line.split()))

    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()
            lines = list(filter(lambda x: len(x) > 0, lines))
        return lines

    def read_crop(self, img_path):
        img_full_path = os.path.join(DATASET_BASE, img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self.crop:
            x1, y1, x2, y2 = self.bbox[img_path]
            if x1 < x2 <= img.size[0] and y1 < y2 <= img.size[1]:
                img = img.crop((x1, y1, x2, y2))
        return img

    def __getitem__(self, index):
        if self.data_type == "train":
            img_path = self.train_list[index]
        elif self.data_type == "val":
            img_path = self.val_list[index]
        else:
            img_path = self.test_list[index]

        img = self.read_crop(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.data_type == "test":
            return img
        else:
            target_attributes = self.attributes[img_path]
            return img, target_attributes


if __name__ == '__main__':
    data = DeepFashionDataset(data_type='train')
