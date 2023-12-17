import math
import cv2
import numpy as np
import random
import copy
from PIL import Image
from .text_image_aug import tia_perspective, tia_stretch, tia_distort
from .rec_img_aug import BaseDataAugmentation
import albumentations as A


class CusAug(object):
    def __init__(self,
                 tia_prob=0.4,
                 crop_prob=0.4,
                 reverse_prob=0.4,
                 noise_prob=0.4,
                 jitter_prob=0.4,
                 blur_prob=0.4,
                 hsv_aug_prob=0.4,
                 **kwargs):
        self.tia_prob = tia_prob
        self.bda = BaseDataAugmentation(crop_prob, reverse_prob, noise_prob,
                                        jitter_prob, blur_prob, hsv_aug_prob)
        self.transform = A.Compose([
            A.Blur(blur_limit=(3, 7), p=0.15),
            A.Defocus(radius=(1, 2), p=0.15),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.ImageCompression(quality_lower=5, quality_upper=100, p=0.2),
            A.JpegCompression(quality_lower=5, quality_upper=100, p=0.2),
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, p=0.1),
            A.RingingOvershoot(blur_limit=(7, 15), p=0.1),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.1),
            A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, always_apply=False, p=0.2),
            A.RandomBrightness(limit=0.2, always_apply=False, p=0.1),
            A.ElasticTransform(alpha=1, sigma=10, alpha_affine=1.5, interpolation=1, border_mode=4, value=None,
                               mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.25),
            A.CoarseDropout(max_holes=8, max_height=100, max_width=20, min_holes=None, min_height=None, min_width=None,
                            fill_value=200, mask_fill_value=None, always_apply=False, p=0.1),
            A.GridDistortion(num_steps=5, distort_limit=0.5, interpolation=1, border_mode=4, value=None,
                             mask_value=None, normalized=False, always_apply=False, p=0.15)
        ])

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape

        # tia
        if random.random() <= self.tia_prob:
            img = tia_distort(img, random.randint(min(w//3, 5), min(w//3, 20)))
            img = tia_stretch(img, random.randint(3, 6))
            img = tia_perspective(img)

        # albumentations
        try:
            transformed = self.transform(image=img)
            img = transformed["image"]
            data['image'] = img
        except Exception as e:
            data['image'] = img
            data = self.bda(data)
            pass

        return data


class BWTransfer(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_image_3_channel = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        data['image'] = gray_image_3_channel
        return data


class RandomShrink(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape
        nw = random.randint(int(w/2), w)
        resized_img = cv2.resize(img, (nw, h))
        data['image'] = resized_img
        return data


class RandomPadding(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        image = data['image']
        height, width = image.shape[:2]
        padding_factor = np.random.uniform(1.0, 1.4)
        new_height = int(height * padding_factor)
        padding_color = [240, 240, 240]
        y_offset = random.randint(0, new_height - height) if new_height > height else 0

        new_image = np.full((new_height, width, 3), padding_color, dtype=np.uint8)
        new_image[y_offset:y_offset + height, 0:width] = image

        data['image'] = new_image
        return data