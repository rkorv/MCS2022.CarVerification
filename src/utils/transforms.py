import albumentations as A
import albumentations.pytorch

import torch


def get_augs_transforms(shape):
    return A.Compose(
        [
            A.Resize(shape[1], shape[0]),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.08, rotate_limit=5, p=0.7),
            A.RandomResizedCrop(
                shape[1], shape[0], scale=(0.9, 1.0), ratio=(0.75, 1.33), interpolation=1, p=0.7
            ),
            A.HorizontalFlip(p=0.5),
            # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.8),
            # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
        ]
    )


def get_final_transforms(shape, mean, std, augs=False):
    return A.Compose(
        [
            A.Resize(shape[1], shape[0]),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            A.pytorch.ToTensorV2(),
        ]
    )


def get_inv_transforms(mean, std):
    return A.Compose(
        [
            A.Normalize(
                mean=[-m / s for m, s in zip(mean, std)],
                std=[1 / s for s in std],
                max_pixel_value=1.0,
            )
        ]
    )


class CarsTransforms:
    def __init__(self, shape=(512, 512), mean=[0, 0, 0], std=[1, 1, 1], augs=False):
        tfs = []
        tfs += [get_augs_transforms(shape)] if augs else []
        tfs += [get_final_transforms(shape, mean, std)]

        self.transforms = A.Compose(tfs)
        self.invert_image = get_inv_transforms(mean, std)

    def denormalize(self, **kwargs):
        if torch.is_tensor(kwargs["image"]):
            kwargs["image"] = kwargs["image"].permute(1, 2, 0).cpu().numpy()

        return self.invert_image(**kwargs)

    def __call__(self, **kwargs):
        return self.transforms(**kwargs)
