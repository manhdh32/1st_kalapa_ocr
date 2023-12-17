import os
import re
import cv2
import numpy as np
import unicodedata
from math import sin, cos
from PIL import Image, ImageFont, ImageDraw


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M


def convert_points(points, image_shape, rotation, padding, scale_factor):
    width, height = image_shape
    center_x, center_y = width // 2, height // 2
    rotation_radians = rotation * (3.14159265358979323846 / 180.0)
    new_points = []
    for x, y in points:
        new_x = (x - center_x) * cos(rotation_radians) + (y - center_y) * sin(rotation_radians) + center_x + padding[0]
        new_y = (y - center_y) * cos(rotation_radians) - (x - center_x) * sin(rotation_radians) + center_y + padding[1]
        new_points.append((int(new_x * scale_factor), int(new_y * scale_factor)))
    return new_points


def random_points(points):
    points[0][0] += np.random.randint(-10, 0)
    points[0][1] += np.random.randint(-10, 0)
    points[1][0] += np.random.randint(0, 10)
    points[1][1] += np.random.randint(-10, 0)
    points[2][0] += np.random.randint(0, 10)
    points[2][1] += np.random.randint(0, 10)
    points[3][0] += np.random.randint(-10, 0)
    points[3][1] += np.random.randint(0, 10)
    return points


def draw_points(image, points):
    draw = ImageDraw.Draw(image)
    draw.polygon(points, outline=(255, 255, 0), width=1)
    return image


def read_icons(folder_path):
    paths = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            fid, ext = os.path.splitext(filename)
            if ext not in [".png", ".jpg", ".jpeg"]:
                continue
            file_path = os.path.join(root, filename)
            paths.append(file_path)
    return paths


def replace_number(text):
    numbers = re.findall(r'\d+', text)
    if numbers:
        for number in numbers:
            rnumber = str(np.random.randint(0, 1000))
            text = text.replace(number, rnumber, 1)
    return text


def rm_dup_space(text):
    return re.sub(' +', ' ', text)


def norm_text(text):
    text = unicodedata.normalize('NFC', text)
    text = text.replace('Ð', 'Đ')
    text = text.replace('’', "'")
    text = ''.join([c for c in text if c in vn_chars])
    text = rm_dup_space(text)
    return text


def random_drop(text):
    ps = [0.01, 0.025, 0.075, 0.15]
    results = []
    parts = text.split(',')
    for idx, part in enumerate(parts):
        ps_idx = len(parts) - idx - 1
        ps_idx = min(ps_idx, len(ps) - 1)
        p = ps[ps_idx] if (len(parts) - idx + len(results)) > 2 else -1
        if np.random.uniform(0.0, 1.0) >= p:
            results.append(part)
    text = ' '.join(results)
    text = rm_dup_space(text).strip()
    return text


def random_delete(text):
    string_list = list(text)
    random_position = np.random.randint(0, len(string_list))
    string_list.pop(random_position)
    new_string = ''.join(string_list)
    return new_string


def random_insert(text):
    position = np.random.randint(0, len(text) + 1)
    p_char1 = text[max(0, position - 1): max(0, position - 1) + 1]
    p_char2 = text[position: position + 1]
    p_char3 = text[min(len(text), position + 1): min(len(text), position + 1) + 1]
    if p_char1.isdigit() or p_char2.isdigit() or p_char3.isdigit():
        return text
    character = np.random.choice(alnums)
    new_string = text[:position] + character + text[position:]
    return new_string


def random_replace(text):
    position = np.random.randint(0, len(text))
    p_char = text[position: position+1]
    if p_char.isdigit():
        return text
    character = np.random.choice(alnums)
    new_string = text[:position] + character + text[position+1:]
    return new_string


def random_bad_text(text):
    ops = [random_insert, random_delete, random_replace]
    num = np.random.randint(1, 5)
    chose_ops = np.random.choice(ops, size=num)
    for op in chose_ops:
        text = op(text)
    return text


with open('resources/vi_chars.txt', 'r') as f:
    lines = f.readlines()
vn_chars = [c.strip() for c in lines]
vn_chars.append(' ')
digits = [c for c in vn_chars if c.isdigit()]
alphas = [c for c in vn_chars if c.isalpha()]
alnums = [c for c in vn_chars if c.isalnum()]