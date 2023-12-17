import json
import os
import re
import time
import sys
import cv2
import copy
import numpy as np
import random
from utils.func_utils import *
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
import albumentations as A
from multiprocessing import Pool
import multiprocessing


def paste_image(src_image, bg_image, opacity=1.0, ratio_range=(0.04, 0.06)):
    imw, imh = src_image.size
    bgW, bgH = bg_image.size

    # paste to random location
    imw, imh = src_image.size
    loc_x = np.random.randint(0, bgW - imw)
    loc_y = np.random.randint(0, bgH - imh)

    mask = (np.asarray(src_image) * opacity).astype(np.uint8)
    mask = Image.fromarray(mask)
    bg_image.paste(src_image, (loc_x, loc_y), mask)

    return bg_image


def replace_number(text):
    numbers = re.findall(r'\d+', text)
    if numbers:
        for number in numbers:
            rnumber = str(random.randint(0, 1000))
            text = text.replace(number, rnumber, 1)
    return text


def create_text_image(text, font, ex_bold=True):
    text_width, text_height = font.getsize(text)
    image = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, fill=(0, 0, 0, np.random.randint(200, 255)), font=font)
    if ex_bold:
        image2 = copy.deepcopy(image)
        image.paste(image2, (1, 0), image2)
    return image


def process_idx(idx):
    np.random.seed(idx)
    dataset_folder = f"{dataset_name}"
    if not os.path.exists(f"train_data/{dataset_name}"):
        try:
            os.mkdir(f"train_data/{dataset_folder}")
            os.mkdir(f"train_data/{dataset_folder}/images")
            os.mkdir(f"train_data/{dataset_folder}/labels")
        except:
            pass
    return gen_image_process(idx, dataset_folder)


def gen_image_process(idx, dataset_folder):
    try:
        # tao text image
        add = np.random.choice(adds).strip()
        text = replace_number(add)
        font = np.random.choice(fonts)

        text_image = create_text_image(text, font, False)
        # rotate
        max_rot = 2
        rotation = np.random.randint(-max_rot, max_rot)
        text_image = text_image.rotate(rotation, expand=True)
        t_width, t_height = text_image.size

        # create bg image
        imw, imh = int(np.random.uniform(1.1, 1.5) * t_width), int(1.2 * t_height)
        image_array = np.full(shape=(imh, imw, 3), fill_value=255).astype(np.uint8)
        image = Image.fromarray(image_array)

        # paste text image vao anh
        image = paste_image(text_image, image, 1.0, (0.8, 0.9))

        image_name = f"{dataset_folder}_{idx}.jpg"
        label_name = f"{dataset_folder}_{idx}.txt"
        image.save(f"train_data/{dataset_folder}/images/{image_name}", quality=75, subsampling=0)
        with open(f"train_data/{dataset_folder}/labels/{label_name}", 'w') as f:
            f.writelines(text)
        outline = f"train_data/{dataset_folder}/images/{image_name}\t{text}"
        return outline
    except Exception as e:
        print(e)
        pass


lines_queue = multiprocessing.JoinableQueue()
# load cac resource can thiet
icons = read_icons("resources/icons")
with open("resources/lm_corpus/root_address.txt", "r") as f:
    adds = f.readlines()

with open('resources/vi_chars.txt', 'r') as f:
    lines = f.readlines()
vn_chars = [c.strip() for c in lines]
vn_chars.append(' ')

# load fonts
font_folder = "resources/fonts"
font_files = os.listdir(font_folder)
fonts = [ImageFont.truetype(f"resources/fonts/{f}", 40) for f in font_files if not f.startswith('.')]

# augmentation
transform = A.Compose([
    A.Blur(blur_limit=(3, 7), p=0.35),
    A.Defocus(radius=(1, 2), p=0.25),
    A.GaussianBlur(blur_limit=(3, 5), p=0.25),
    A.GlassBlur(sigma=0.2, max_delta=1, p=0.1),
    A.ZoomBlur(max_factor=(1.0, 1.15), p=0.1),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.ImageCompression(quality_lower=5, quality_upper=100, p=0.5),
    A.JpegCompression(quality_lower=5, quality_upper=100, p=0.5),
    A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.3, p=0.1),
    A.RingingOvershoot(blur_limit=(7, 15), p=0.1),
    A.Spatter(p=0.1),
])


with open("resources/address_corpus.txt", "r") as f:
    adds = f.readlines()
num_image = 1000000
dataset_name = "synthetic_train"
with Pool(32) as p:
    outlines = list(tqdm(p.imap(process_idx, range(num_image)), total=num_image))
with open(f"train_data/{dataset_name}/labels.txt", 'w') as f:
    for line in outlines:
        if line is not None:
            f.writelines(line + '\n')

num_image = 5000
dataset_name = "synthetic_valid"
with Pool(32) as p:
    outlines = list(tqdm(p.imap(process_idx, range(num_image)), total=num_image))
with open(f"train_data/{dataset_name}/labels.txt", 'w') as f:
    for line in outlines:
        if line is not None:
            f.writelines(line + '\n')
