import sys
sys.path.append('./mmocr')
sys.path.append('./parseq')

from mmocr.apis.inferencers import MMOCRInferencer
import cv2
import numpy as np
import torch
import os
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args, get_pretrained_weights
import time
import gc

# Report speed of detection
DET_DATA = 'vietnamese/unseen_test_images'
init_args = {
    'det': 'mmocr/configs/textdet/dbnetpp/dbnetpp_resnet18-dcnv2_fpnc_1200e_aic2021.py', 
    'det_weights': 'pretrained/dbnetpp_resnet18-dcnv2_fpnc_1200e_icdar2015_20220829_230108-f289bd20.pth', 
    'rec': None, 
    'rec_weights': None, 
    'kie': None, 
    'kie_weights': None,
    'device': 'cuda'
}

detector = MMOCRInferencer(**init_args)
start = time.time()
img_names = os.listdir(DET_DATA)
n_imgs = len(img_names) * 2

for img_name in img_names:
    img_path = os.path.join(DET_DATA, img_name)
    img = cv2.imread(img_path)
    res = detector(img, batch_size=1)
    print(res)
    break

import sys
sys.exit()

print(f'Speed of detection: {np.round((time.time() - start) / n_imgs, 4)} second per image') 
del detector; gc.collect()

# Report speed of recognizer
RECOG_DATA = 'vietnamese/recog-data'
img_names = os.listdir(RECOG_DATA)[:200]
n_imgs = len(img_names)
recognizer = load_from_checkpoint('pretrained=parseq').eval().to('cuda')
img_transform = SceneTextDataModule.get_transform(recognizer.hparams.img_size)
batch_size = 10

start = time.time()
for i in range(0, n_imgs, batch_size):
    img_batch = []
    for img_name in img_names[i * batch_size: (i + 1) * batch_size]:
        img = Image.open(os.path.join(RECOG_DATA, img_name))
        image = img_transform(img).unsqueeze(0)
        img_batch.append(image)
    
    if len(img_batch) == 0:
        continue

    img_batch = torch.cat(img_batch).to('cuda')
    probs = recognizer(img_batch).softmax(-1)
    preds, probs = recognizer.tokenizer.decode(probs)

print(f'Speed of recognizer: {np.round((time.time() - start) / n_imgs, 4)} (second per image)')



