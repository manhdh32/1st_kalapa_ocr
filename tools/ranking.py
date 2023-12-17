import os
import time
import sys
import cv2
import csv
import Levenshtein
import numpy as np
from PIL import Image
from tqdm import tqdm
from math import pow
from ocr.ocr_any import OcrAny


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


def one_checkpoint():
    exp_name = 'final_kalapa'
    root = f"PaddleOCR/output/{exp_name}"
    cp_files = [file for file in os.listdir(root) if (file.endswith('.pdparams'))]
    outfile = open(f"submission/{exp_name}.csv", 'w')
    outfile.writelines(f'fid\tstep\tscore\tchar_score\tword_score\tacc\n')
    for idx, file in enumerate(cp_files):
        print(f"------------------{idx + 1}/{len(cp_files)}. {file}--------------------------")
        fid = os.path.splitext(file)[0]
        if file.startswith('step'):
            _, step, score = fid.split('_')
        elif file.startswith("iter_"):
            _, _, epoch = fid.split('_')
            step = f"epoch_{epoch}"
            score = 0
        else:
            continue

        if not os.path.exists(f"PaddleOCR/output/{exp_name}/files/{fid}"):
            os.system(
                f"python PaddleOCR/tools/export_model.py -c PaddleOCR/output/{exp_name}/config.yml -o Global.pretrained_model=PaddleOCR/output/{exp_name}/{fid} Global.save_inference_dir=PaddleOCR/output/{exp_name}/files/{fid}")
        if not os.path.exists(f"PaddleOCR/output/{exp_name}/files/{fid}.onnx"):
            command = f"paddle2onnx --model_dir PaddleOCR/output/{exp_name}/files/{fid}" + """ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file""" + f" PaddleOCR/output/{exp_name}/files/{fid}.onnx" + """ --opset_version 12 --input_shape_dict="{'x':[-1,3,48,-1]}" --enable_onnx_checker True"""
            os.system(command)

        onnx_path = f"PaddleOCR/output/{exp_name}/files/{fid}.onnx"
        print(onnx_path)
        char_score, acc = test(onnx_path)
        outfile.writelines(f'{fid}\t{step}\t{score}\t{char_score}\t{acc}\n')

    outfile.close()


if __name__ == '__main__':
    one_checkpoint()