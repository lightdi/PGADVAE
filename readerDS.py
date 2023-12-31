# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label, disguise, proPath = line.strip().split(' ')
            if int(disguise) != 0:
                disguise = 1 
            imgList.append((imgPath, int(label), int(disguise), proPath))
    return imgList

class FaceIdExpDataset(Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        
    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgPath, id_label, disguise_label, proPath = self.imgList[idx]
        try:
            img = Image.open(self.root+ imgPath + '.bmp')
        except:
            print(self.root+ imgPath)
        img = img.convert('L').convert('RGB')
        img= self.transform(img)
        try:
            pro = Image.open(self.root + proPath + '.bmp')
        except:
            print(self.root+ imgPath)
        pro = pro.convert('L').convert('RGB')
        pro= self.transform(pro)
        return [img.float(), id_label, disguise_label, pro.float()]
#        return [os.path.join(self.root, imgPath), id_label, exp_label, pro.float()]


def get_batch(root, fileList, batch_size, shuffle=True):
    data_set = FaceIdExpDataset(root, fileList,
                                 transform=transforms.Compose([
                                     transforms.Resize((64,64)),
#                                     transforms.CenterCrop((96,96)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                 ]))
    dataloader = DataLoader(data_set,batch_size=batch_size,
                            shuffle=shuffle,drop_last=True, num_workers=6)  #drop_last is necessary,because last iteration may fail 
    return dataloader

if __name__=='__main__':
    data_set = get_batch('./PEAL_data','dataset/LoadPEAL200.txt',16)
    for i, data_batch in enumerate(data_set):
        img = data_batch[0]
        ID = data_batch[1]
        exp = data_batch[2]
        pro = data_batch[3]
        print(exp.data.numpy())