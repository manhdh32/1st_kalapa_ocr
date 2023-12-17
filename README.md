# 1st Kalapa ByteBattles 2023  - Vietnamese Handwritten Text Recognition

## 1.Installation

``` bash
- git clone https://github.com/manhdh32/1st_kalapa_ocr.git
- cd 1st_kalapa_ocr
- pip install -r requirements.txt
```

## 2. Data

- Kalapa [dataset](https://drive.google.com/drive/folders/1s3mGm31XuI5v8Q2__-Y5m_9vZZQXtqwI?usp=drive_link).
- Vietnamese address [dataset](https://github.com/thien0291/vietnam_dataset)
- Processed [address corpus](https://github.com/manhdh32/1st_kalapa_ocr/blob/main/resources/address_corpus.txt)
- Synthetic data. Download my generated data [here](https://drive.google.com/drive/folders/1B2D5eh3euxtOKAPUJiH3jacatfm4UHKh?usp=sharing)

## 3. Training
Get pretrained model [here](https://drive.google.com/drive/folders/1v4k_JIBwv008quUagvS5bMOgGzTJVmFS?usp=drive_link) or generate synthetic data with:

``` bash
python tools/gen_synthetic_data.py
```
and training from scratch:
``` bash
python PaddleOCR/tools/train.py -c configs/pretrained_config.yml
```

Fine-tune:
``` bash
python PaddleOCR/tools/train.py -c configs/final_kalapa.yml
```

## 4. Inference
Inference notebook [here](https://www.kaggle.com/domanh/h2h-notebook-01)