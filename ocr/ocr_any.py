import copy
import re
import json
import cv2
import math
import numpy as np
import onnxruntime
from pyctcdecode import build_ctcdecoder
from paddle import inference
import time


class OcrAny:
    def __init__(self,
                 text_rec_path: str = None,
                 providers: list = ['CPUExecutionProvider'],
                 character_dict_path: str = None,
                 use_lm: bool = True,
                 lm_path: str = None,
                 alpha: float = 0.5,
                 beta: float = 1.0):
        self.text_rec_path = text_rec_path
        if text_rec_path.endswith('onnx'):
            self.use_onnx = True
            self.text_rec = onnxruntime.InferenceSession(text_rec_path, providers=providers)
            self.rec_output_names = [x.name for x in self.text_rec.get_outputs()]
        else:
            self.use_onnx = False
            self.enable_mkldnn = False
            self.predictor, self.input_tensor, self.output_tensors, self.config = self.create_predictor()
        self.min_size = 3
        self.beg_str = "sos"
        self.end_str = "eos"
        self.character_str = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        self.character_str.append(" ")
        dict_character = list(self.character_str)
        dict_character = ['blank'] + dict_character
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        new_dict = copy.deepcopy(self.character)
        new_dict[0] = ''
        self.use_lm = use_lm
        if self.use_lm:
            self.decoder = build_ctcdecoder(
                new_dict,
                kenlm_model_path=lm_path,
                alpha=alpha,
                beta=beta,
            )

    def get_model_outputs(self, norm_img_batch):
        norm_img_batch = norm_img_batch.copy()
        if self.use_onnx:
            rec_input_dict = {'x': norm_img_batch}
            textrec_outputs = self.text_rec.run(self.rec_output_names, rec_input_dict)
        else:
            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.run()
            textrec_outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                textrec_outputs.append(output)
        return textrec_outputs

    def predict_single(self, image, use_lm=True):
        norm_img_batch = self.rec_preprocess(image)
        textrec_outputs = self.get_model_outputs(norm_img_batch)
        text_preds = textrec_outputs[0]
        if use_lm:
            text, score = self.decode_with_lm(preds=text_preds)[0]
        else:
            text, score = self.decode(text_preds)[0]
        return text, score

    def predict(self, image):
        norm_img_batch = self.rec_preprocess(image)
        textrec_outputs = self.get_model_outputs(norm_img_batch)
        text_preds = textrec_outputs[0]
        text, score1 = self.decode(text_preds)[0]
        if score1 < 0.7 and len(text) < 8:
            return "", score1
        text_lm, score2 = self.decode_with_lm(preds=text_preds)[0]
        words = text.split(' ')
        words_lm = text_lm.split(' ')
        if len(words) != len(words_lm):
            return text_lm, score1
        for idx, word in enumerate(words):
            if re.search(r'\d', word) and word != words_lm[idx]:
                words_lm[idx] = word
        text = ' '.join(words_lm)
        return text, score1

    def predict_compare(self, image):
        norm_img_batch = self.rec_preprocess(image)
        textrec_outputs = self.get_model_outputs(norm_img_batch)
        text_preds = textrec_outputs[0]
        text, score1 = self.decode(text_preds)[0]
        text_lm, score2 = self.decode_with_lm(preds=text_preds)[0]
        return text, text_lm

    def rec_preprocess(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        norm_img = self.resize_norm_img(image)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch = [norm_img]
        norm_img_batch = np.concatenate(norm_img_batch)
        return norm_img_batch

    def decode(self, preds, is_remove_duplicate=True):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        result_list = []
        ignored_tokens = [0]
        batch_size = len(preds_idx)
        for batch_idx in range(batch_size):
            selection = np.ones(len(preds_idx[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = preds_idx[batch_idx][1:] != preds_idx[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= preds_idx[batch_idx] != ignored_token
            char_list = [
                self.character[text_id]
                for text_id in preds_idx[batch_idx][selection]
            ]
            if preds_prob is not None:
                conf_list = preds_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
            text = ''.join(char_list)
            text = text.replace('✓', '')

            score = np.mean(conf_list).tolist()
            if score < 0.7 and len(text) < 8:
                text = ''
            result_list.append((text, score))
        return result_list

    def decode_with_lm(self, preds):
        result = self.decoder.decode(preds[0], beam_width=50)
        result = result.replace('✓', '')
        return [(result, 1.0)]

    def create_predictor(self):
        model_file_path = f'{self.text_rec_path}/inference.pdmodel'
        params_file_path = f'{self.text_rec_path}/inference.pdiparams'
        config = inference.Config(model_file_path, params_file_path)
        config.disable_gpu()
        if self.enable_mkldnn:
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(10)
        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.delete_pass("matmul_transpose_reshape_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)
        predictor = inference.create_predictor(config)
        input_names = predictor.get_input_names()
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
        output_tensors = self.get_output_tensors(predictor)
        return predictor, input_tensor, output_tensors, config

    def get_output_tensors(self, predictor):
        output_names = predictor.get_output_names()
        output_tensors = []
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
        return output_tensors

    def set_enable_mkldnn(self, is_enable):
        self.enable_mkldnn = is_enable
        self.predictor, self.input_tensor, self.output_tensors, self.config = self.create_predictor()

    def warp_up(self):
        if not self.use_onnx:
            img = np.random.randint(0, 255, size=(103, 1400, 3), dtype=np.uint8)
            start = time.time()
            self.set_enable_mkldnn(False)
            self.predict(img)
            time1 = time.time() - start
            start = time.time()
            try:
                self.set_enable_mkldnn(True)
                self.predict(img)
                time2 = time.time() - start
                if time2 > time1:
                    self.set_enable_mkldnn(False)
            except Exception as e:
                print(f"Error in enable mkldnn: {e}")
                self.set_enable_mkldnn(False)

    @staticmethod
    def resize_norm_img(img):
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
        # padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im = np.zeros((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, 0:resize_w] = resized_image
        return padding_im

