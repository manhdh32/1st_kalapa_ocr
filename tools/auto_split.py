import os
import cv2
import math
import numpy as np
import onnxruntime
import pickle as pkl
from tqdm import tqdm


def local_resize_norm_img(img):
    width_downsample_ratio = 0.25
    image_shape = (3, 48, 48, 720)
    imgC, imgH, imgW_min, imgW_max = image_shape
    h = img.shape[0]
    w = img.shape[1]
    # make sure new_width is an integral multiple of width_divisor.
    width_divisor = int(1 / width_downsample_ratio)
    # resize
    ratio = w / float(h)
    resize_w = math.ceil(imgH * ratio)
    if resize_w % width_divisor != 0:
        resize_w = round(resize_w / width_divisor) * width_divisor
    if imgW_min is not None:
        resize_w = max(imgW_min, resize_w)
    if imgW_max is not None:
        valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
        resize_w = min(imgW_max, resize_w)
    resized_image = cv2.resize(img, (resize_w, imgH))
    resized_image = resized_image.astype('float32')
    # norm
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
    padding_im[:, :, 0:resize_w] = resized_image
    return padding_im, resize_w/w, resize_w


if __name__=='__main__':
    text_rec_path="onnxs/kalapa_from_synthesis_fixed_data.onnx"
    character_dict_path='resources/vi_chars_fixed.txt'
    with open(character_dict_path, 'r') as f:
        chars = f.readlines()
        char2idx = {c.replace('\n', ''): idx for (idx, c) in enumerate(chars)}
    text_rec = onnxruntime.InferenceSession(text_rec_path, providers=['CPUExecutionProvider'])
    rec_output_names = [x.name for x in text_rec.get_outputs()]
    root = 'train_data/kalapa_valid_fixed/images'
    root_label = 'train_data/kalapa_valid_fixed/labels'
    files = os.listdir(root)

    word_data = {}
    for file in tqdm(files):
        if file.startswith('.'):
            print("Continue file ", file)
            continue
        filepath = f"{root}/{file}"
        fid, _ = os.path.splitext(file)
        with open(f'{root_label}/{fid}.txt', 'r') as f:
            label = f.readlines()[0].strip()
        img = cv2.imread(filepath)

        norm_image, ratio, padding = local_resize_norm_img(img)
        norm_image = norm_image[np.newaxis, :]
        norm_img_batch = [norm_image]
        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()
        rec_input_dict = {'x': norm_img_batch}
        textrec_outputs = text_rec.run(rec_output_names, rec_input_dict)
        text_preds = textrec_outputs[0]
        preds_idx = text_preds.argmax(axis=2)
        spaces = []
        start = False
        first_non_zero_index = np.argmax(preds_idx[0] != 0)
        last_non_zero_index = len(preds_idx[0]) - 1 - np.argmax(preds_idx[0][::-1] != 0)
        for idx, char_id in enumerate(preds_idx[0]):
            if char_id == 231:
                if not start:
                    spaces.append([])
                start = True
                spaces[-1].append(idx)
            else:
                start = False
        spaces = [[max(0, first_non_zero_index - 2)]] + spaces + [[min(180, last_non_zero_index + 5)]]
        if len(label.split()) != (len(spaces) - 1):
            print(label)
            continue
        splits = []
        last = None
        for sps in spaces:
            if last is None:
                last = np.mean(sps)*720/180
                continue
            splits.append((last, (np.mean(sps) + 1)*720/180))
            last = np.mean(sps)*720/180

        words = label.split()
        assert len(words) == len(splits)
        for idx, word in enumerate(words):
            start, end = splits[idx]
            start = int(start/ratio)
            end = int(end/ratio)
            word_img = img[:, start: end, :]
            h, w, _ = word_img.shape
            if h*w*_ == 0:
                continue

            # result, score = ocr_model.predict(word_img)
            # if word != result.strip():
            #     print(f"Word: {word}. Pred: {result}")
            #     continue

            bw_word_img = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
            h, w = bw_word_img.shape
            ratio_scale = 32/h
            nw, nh = int(ratio_scale*w), int(ratio_scale*h)
            if nw * nh == 0:
                continue
            bw_word_img = cv2.resize(bw_word_img, (32, nh))
            # cv2.imshow("temp", bw_word_img)
            # cv2.waitKey()
            word_data[f"{fid}_{idx}"] = [[char2idx[char] for char in word], bw_word_img]

    print(f'Number of images = {len(word_data)}')
    # Save the data
    with open(f'train_data/splited_words/kalapa_words_valid.pkl', 'wb') as f:
        pkl.dump({'word_data': word_data,
                  'char_map': char2idx,
                  'num_chars': len(char2idx.keys())}, f, protocol=pkl.HIGHEST_PROTOCOL)
