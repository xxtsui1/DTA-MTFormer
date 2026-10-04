import os
import torch
import numpy as np
import random
import random
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from scipy import ndimage
from skimage import data,filters,feature
from scipy.ndimage import distance_transform_edt as distance
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

#import matplotlib.pyplot as plt

# from transform1 import randonm_resize, random_rotate, rotate_resize


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, root, list_file, input_size, state):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.fnames = []
        self.input_size=input_size
        self.state=state
        if state =='Train':
            self.istrain= True
        else:
            self.istrain= False
        

        with open(list_file) as f:
            lines = f.readlines()
        for line in lines:
            self.fnames.append(line[:-1])
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.

        idx = idx % self.num_samples
        fname1 = self.fnames[idx].split(' ')[0]
        cls_fname1 = int(self.fnames[idx].split(' ')[-1])
        if cls_fname1 == 1:
            img = Image.open(os.path.join(self.root[0], fname1))
            mask = Image.open(os.path.join(self.root[1], fname1))
            mask = mask.convert('L')
            mask = np.array(mask)
            mask = (mask>0)*255
            mask = np.uint8(np.interp(mask, (mask.min(), mask.max()), (0, 255)))

        else:
            img = Image.open(os.path.join(self.root[2], fname1))
            mask = np.zeros(img.size, np.uint8)
        #######################################################################
        mask = Image.fromarray(mask,mode='L')
        img,mask = self.build_transform(self.istrain,img,mask,self.input_size) 
        return img,cls_fname1,mask
    
    def build_transform(self,is_train,img,mask, input_size):
        resize_im = input_size > 32
        image=img
        
        if is_train:
            translater = transforms.RandomAffine(5, translate=(0, 0.1), scale=(0.9, 1.1), shear=0)
            angle, translations, scale, shear = translater.get_params(translater.degrees, translater.translate,
                                                                  translater.scale, translater.shear, img.size)
            image = TF.affine(image, angle, translations, scale, shear )
            mask = TF.affine(mask, angle, translations, scale, shear )
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)      
        t=[]
        if resize_im:#if resize_im and not args.gen_attention_maps:
            size = int((256 / 224) * input_size)
            t.append(
                transforms.Resize((size,size), interpolation=0),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(input_size))
        t.append(transforms.ToTensor())
        trans =transforms.Compose(t)
        image = trans(image)
        mask = trans(mask)
        return image,mask

    def _img_transform(self, img):
        return np.array(img)

    def _index_transform(self, index):
        return torch.LongTensor(np.array(index).astype('float32'))#

    def __len__(self):
        return self.num_samples


