import os
import random
import math

import pandas as pd
import cv2
import numpy as np

import torch.utils.data as data


def read_image(path):
    if os.path.isfile(path):
        return (cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
    else:
        print("[ERROR] File not exist: ", path)
        return np.zeros((100, 100, 3), dtype=np.float32)


class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if np.random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                    img[1, x1 : x1 + h, y1 : y1 + w] = self.mean[1]
                    img[2, x1 : x1 + h, y1 : y1 + w] = self.mean[2]
                else:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                return img

        return img


class CarsDatasetBase(data.Dataset):
    def __init__(
        self,
        imgs_root,
        annotation_file,
        transforms=None,
        inference=False,
        triplet=False,
        augs=False,
        class_offset=0,
    ):
        self.imgs_root = imgs_root
        self.annotation_file = annotation_file
        self.triplet = triplet
        self.augs = augs

        self.imlist = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.inference = inference
        self.class_offset = class_offset
        if not self.inference:
            self.imlist["class"] += self.class_offset

        if self.augs:
            self.eraser = RandomErasing(mean=[0.0, 0.0, 0.0])

        cv2.setNumThreads(6)

    def class_count(self):
        return self.imlist["class"].max() + 1

    def _load_img(self, sample):
        impath, x1, y1, x2, y2 = sample[:5]

        full_imname = os.path.join(self.imgs_root, impath)
        if not os.path.exists(full_imname):
            print("No file ", full_imname)

        img = read_image(full_imname)

        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            x1, x2, y1, y2 = (
                x1 * img.shape[1],
                x2 * img.shape[1],
                y1 * img.shape[0],
                y2 * img.shape[0],
            )

        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))

        if 0 <= x1 < x2 and 0 <= y1 < y2 and 0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]:
            if self.augs:
                scale = 0.2
                w = x2 - x1
                h = y2 - y1
                w_scale = (w * np.random.rand(2) * (scale / 2)).astype(np.int)
                h_scale = (h * np.random.rand(2) * (scale / 2)).astype(np.int)

                x1 = max(x1 - w_scale[0], 0)
                x2 = min(x2 + w_scale[1], img.shape[1] - 1)
                y1 = max(y1 - h_scale[0], 0)
                y2 = min(y2 + h_scale[1], img.shape[0] - 1)

            img = img[y1:y2, x1:x2]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        if self.augs:
            img = self.eraser(img)

        return img

    def get_negative_sample(self, target_sample):
        target_class = target_sample.values[5]
        other_classes = self.imlist[self.imlist["class"] != target_class]
        return other_classes.sample().values[0]

    def get_triplet_samples(self, index):
        target_sample = self.imlist.loc[index]
        target_class = target_sample.values[5]

        same_class = self.imlist[self.imlist["class"] == target_class]

        if len(same_class) > 1:
            positive_sample = same_class[same_class.index != index].sample().values[0]
        else:
            positive_sample = target_sample.values

        negative_sample = self.get_negative_sample(target_sample)

        return target_sample, positive_sample, negative_sample

    def get_triplet_images(self, index):
        target_sample, positive_sample, negative_sample = self.get_triplet_samples(index)

        target_img = self._load_img(target_sample)
        positive_img = self._load_img(positive_sample)
        negative_img = self._load_img(negative_sample)

        return (
            [target_img, target_sample[5]],
            [positive_img, positive_sample[5]],
            [negative_img, negative_sample[5]],
        )

    def __getitem__(self, index):
        return self.get_image_item(index)

    def get_image_item(self, index):
        if self.triplet:
            return list(zip(*self.get_triplet_images(index)))

        sample = self.imlist.loc[index].values

        img = self._load_img(sample)

        if self.inference:
            return img
        else:
            target_class = sample[5]
            return img, target_class

    def __len__(self):
        return len(self.imlist)


class CarsDatasetInference(CarsDatasetBase):
    def __init__(self, root, transforms=None, scale=None):
        ds_path = os.path.join(root, "public_test")

        annotation_file = os.path.join(ds_path, "images_w_boxes.csv")
        imgs_root = os.path.join(ds_path, "images")
        self.infernece_scale = scale

        super().__init__(imgs_root, annotation_file, transforms=transforms, inference=True)

    def _load_img(self, sample):
        impath, x1, y1, x2, y2 = sample[:5]

        full_imname = os.path.join(self.imgs_root, impath)
        if not os.path.exists(full_imname):
            print("No file ", full_imname)

        img = read_image(full_imname)

        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))

        if 0 <= x1 < x2 and 0 <= y1 < y2 and 0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0]:
            if self.infernece_scale is not None:
                scale = self.infernece_scale
                w = x2 - x1
                h = y2 - y1
                w_scale = w * (scale / 2)
                h_scale = h * (scale / 2)

                x1 = int(max(x1 - w_scale, 0))
                x2 = int(min(x2 + w_scale, img.shape[1] - 1))
                y1 = int(max(y1 - h_scale, 0))
                y2 = int(min(y2 + h_scale, img.shape[0] - 1))

            img = img[y1:y2, x1:x2]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return img


class FullDatasetBalanced(CarsDatasetBase):
    def __init__(self, root, split="full", **kwargs):
        ds_path = os.path.join(root, "vehicle_models")

        imgs_root = os.path.join(ds_path, "images")
        merged_file = os.path.join(ds_path, "train.csv")

        super().__init__(imgs_root, merged_file, **kwargs)
        FullDatasetBalanced.CLASS_NUM = super().class_count()

    def get_negative_sample(self, target_sample):
        target_class = target_sample.values[5]

        target_type = target_sample.values[6]
        if target_type == "car":
            if random.random() < 0.8:
                search_mask = self.imlist["parent"] == "car"
            else:
                search_mask = self.imlist["parent"] != "car"
        else:
            if random.random() < 0.8:
                search_mask = self.imlist["parent"] != "car"
            else:
                search_mask = self.imlist["parent"] == "car"

        other_classes = self.imlist[(self.imlist["class"] != target_class) & search_mask]

        return other_classes.sample().values[0]
