import numpy as np
from torch.utils.data import Dataset
import cv2
import torch

def random_crop(img, crop_h, crop_w):
    """
        img: an array in the shape of HxWxC, 0-255, integer
        return an image in the shape of crop_h x crop_w x C
    """
    h,w = img.shape[:2]
    
    max_h = h - crop_h + 1
    max_w = w - crop_w + 1

    
    start_h = np.random.randint(0, max_h)
    start_w = np.random.randint(0, max_w)
    
    return img[start_h : start_h+crop_h, start_w : start_w+crop_w, :]


def center_crop(img):
    
    """
        img: an array in the shape of HxWxC, 0-255, integer
        return an image in the shape of (HxHxC) if W>H or (WxWxC) if H>W
    """
    
    h,w = img.shape[:2]
    if h > w:
        start = (h-w) // 2
        end = -start
        result = img[start:end, :, :]
    elif w > h:
        start = (w-h) // 2
        end = -start
        result = img[:, start:end, :]
    
    return result
    

class custom_dataset(Dataset):
    
    def __init__(self, config, json_file, split):
        self.config = config
        self.split = split
        self.imdb = json_file[f'{split}_data']
        self.img_h = config['img_h']
        self.img_w = config['img_w']
        self.classes = []
        
        for data in self.imdb:
            if data['digit_name'] not in self.classes:
                self.classes.append(data['digit_name'])
        
        self.classes = sorted(self.classes)
        
    
    def __len__(self):
        return len(self.imdb)
        
    
    def __getitem__(self, index):
        
        # Obtain the entry
        entry = self.imdb[index]
        
        # Obtain the digit class
        digit_class = int(entry['digit_name'])
        
        # Obtain the image
        # 1. Obtain the digit image
        digit_img = cv2.imread(entry['digit_image']) # HxWxC, BGR, numpy, 0-255, integer, 28x28
        digit_img = cv2.resize(digit_img, (self.img_h, self.img_w)) # HxWxC, BGR, numpy, 0-255, integer, 224x224
        digit_img = digit_img[:, :, [2,1,0]] # HxWxC, RGB, numpy, 0-255, integer, 224x224
        
        # Discretize it into either 255 or 0 for better output
        digit_img[digit_img <= 50] = 0
        digit_img[digit_img > 50] = 255 # HxWxC, RGB, numpy, 0 or 255, integer, 224x224
        
        # 1.1 Obtain the mask
        mask = digit_img.copy()
        mask[mask <=50] = 0
        mask[mask >50] = 1

        
        # 1.2 Apply color

        digit_img = np.concatenate((digit_img[:, :, 0][..., None] * float(entry['color_r']),
                                   digit_img[:, :, 1][..., None] * float(entry['color_g']),
                                   digit_img[:, :, 2][..., None] * float(entry['color_b'])), axis=-1)

        
        # 2. Obtain the background image
        bg_img = cv2.imread(entry['texture_image']) # HxWxC, BGR, numpy, 0-255, integer
        bg_img = bg_img[:, :, [2,1,0]] # HxWxC, RGB, numpy, 0-255, integer
        
        if self.split == 'train':
            bg_img = random_crop(img = bg_img,
                                 crop_h = self.img_h,
                                 crop_w = self.img_w) # HxWxC, RGB, numpy, 0-255, integer, 224X224
        else:
            bg_img = center_crop(img = bg_img)
            bg_img = cv2.resize(bg_img, (self.img_h, self.img_w)) # HxWxC, RGB, numpy, 0-255, integer, 224X224
         
        # 3. Multiply the 2 image
        output = mask*digit_img + (1-mask) * bg_img # HxWxC, RGB, numpy, 0-255, integer, 224X224
        
        # 4. Return img and digit_class
        output = torch.tensor(output).float() # HxWxC, RGB, tensor, 0-255, float, 224X224
        output = output / 255 # HxWxC, RGB, tensor, 0 - 1, float, 224X224
        output = (output * 2) - 1# HxWxC, RGB, tensor, -1 to 1, float, 224X224
        output = output.permute(2,0,1) # CxHxW, RGB, tensor, -1 to 1, float, 224X224
        
        return output, digit_class
        
        
        
    
    