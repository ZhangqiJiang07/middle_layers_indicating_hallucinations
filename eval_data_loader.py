'''
Copied from: https://github.com/LALBJ/PAI/blob/master/eval_data_loader.py
'''

import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class COCODataSet(Dataset):
    def __init__(self, data_path, trans):
        self.data_path = data_path
        self.trans = trans

        img_files = os.listdir(self.data_path)
        random.shuffle(img_files)
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_id = int(img_file.split(".jpg")[0][-6:])

        image = Image.open(os.path.join(self.data_path, img_file)).convert("RGB")
        
        image = self.trans(image, return_tensors="pt")

        return {"img_id": img_id, "image": image}