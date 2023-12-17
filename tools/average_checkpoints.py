import os
import time
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
import paddle
import paddle.nn as nn
import copy
from itertools import combinations
import cv2
import Levenshtein
import numpy as np
from PIL import Image
from tqdm import tqdm
from math import pow
from ocr.ocr_any import OcrAny
import shutil


# import torch


def norm(text):
    text = text.replace(" ", '').strip()
    return text


def test(onnx_path):
    root = "train_data/OCR/level_23"
    root_images = f"{root}/images"
    root_labels = f"{root}/labels"

    ocr_model = OcrAny(text_rec_path=onnx_path,
                       providers=['CUDAExecutionProvider'],
                       character_dict_path='resources/vi_chars.txt',
                       lm_path=f'final_address.arpa',
                       use_lm=True,
                       alpha=0.5,
                       beta=1.0)
    total = 0
    true = 0
    char_score = 0
    start = time.time()
    files = os.listdir(root_images)
    for file in tqdm(files):
        if file.startswith('.'):
            continue
        fid = file.split('.')[0]
        with open(f"{root_labels}/{fid}.txt", 'r') as f:
            lines = f.readlines()
            label = lines[0] if len(lines) > 0 else ''
            label = norm(label)
        img_path = f'{root_images}/{file}'
        img = cv2.imread(img_path)

        result, score = ocr_model.predict(img)
        result = norm(result)
        total += 1
        if result == label:
            true += 1
            cscore = 1
        else:
            if label != '':
                ced = Levenshtein.distance(result, label)
                wed = Levenshtein.distance(result.split(), label.split())
                cscore = max(0, 1 - pow(2.0, ced) / len(label))
            else:
                cscore = 0
        char_score += cscore
    print(f"{char_score}/{total}. Char Score: {char_score / total}")
    print(f"Acc1: {true / total} with time: {time.time() - start}")
    return char_score / total, true / total


def auto_average(root):
    mdict = []
    mfiles = [f for f in os.listdir(root) if f.endswith('.pdparams')]
    for file in mfiles:
        if file.startswith('step'):
            _, step, score = file.replace('.pdparams', '').split('_')
        elif file.startswith('iter'):
            _, _, epoch = file.replace('.pdparams', '').split('_')
            step = f"epoch{epoch}"
        else:
            print(file)
            continue
        static_dict = paddle.load(f"{root}/{file}")
        mdict.append({'step': step, 'static_dict': static_dict})

    for i in range(2, 6, 1):
        list_combine = list(combinations(mdict, i))
        for combine in list_combine:
            new_state_dict = copy.deepcopy(combine[0]['static_dict'])
            fid = '_'.join([d['step'] for d in combine])
            output_path = f'{root}/temp/{fid}.pdparams'
            if os.path.exists(output_path):
                yield output_path
                continue
            for key in new_state_dict:
                current_dict_list = [d['static_dict'][key] for d in combine]
                new_state_dict[key] = sum(current_dict_list) / len(current_dict_list)
            # Lưu static_dict trung bình ra file .pdparams
            paddle.save(new_state_dict, output_path)
            yield output_path


def perform_average(root, exp_name):
    outfile = open(f"onnxs/{exp_name}/temp/{exp_name}_average.csv", 'w')
    outfile.writelines(f'fid\tfid\tchar_score\tacc\n')
    for output_path in auto_average(root):
        folder, filename = os.path.split(output_path)
        fid, _ = os.path.splitext(filename)
        if not os.path.exists(f"{folder}/{fid}"):
            os.system(
                f"python PaddleOCR/tools/export_model.py -c PaddleOCR/output/{exp_name}/config.yml -o Global.pretrained_model={folder}/{fid} Global.save_inference_dir={folder}/{fid}")
        onnx_path = f"{folder}/{fid}.onnx"
        if not os.path.exists(f"{folder}/{fid}.onnx"):
            command = f"paddle2onnx --model_dir {folder}/{fid}" + """ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file""" + f" {folder}/{fid}.onnx" + """ --opset_version 12 --input_shape_dict="{'x':[-1,3,48,-1]}" --enable_onnx_checker True"""
            os.system(command)
        print(onnx_path)
        char_score, acc = test(onnx_path)
        outfile.writelines(f'{fid}\t{fid}\t{char_score}\t{acc}\n')
    outfile.close()


if __name__ == "__main__":
    exp_name = 'final_kalapa'
    checkpoints = ['step_150480_0.9950773324724861', 'step_176814_0.9950877055584926', 'iter_epoch_370',
                   'step_158004_0.995248124538684', 'iter_epoch_390', 'iter_epoch_320']
    for checkpoint in checkpoints:
        src = f"PaddleOCR/output/{exp_name}/files/{checkpoint}.onnx"
        dst = f"onnxs/{exp_name}/{checkpoint}.onnx"
        shutil.copy(src, dst)
        src = f"PaddleOCR/output/{exp_name}/{checkpoint}.pdparams"
        dst = f"onnxs/{exp_name}/{checkpoint}.pdparams"
        shutil.copy(src, dst)

    root = f'onnxs/{exp_name}'
    perform_average(root, exp_name)