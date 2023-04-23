import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from kaggle.api.kaggle_api_extended import KaggleApi

class DrivingDataset(Dataset):
    def __init__(self):
        self.download_dataset()
        self.data = self.load_data()

    def download_dataset(self):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('asrsaiteja/car-steering-angle-prediction', path='input')

    def load_data(self):
        folder = './input/driving_dataset/'
        data_list=[]
        with open(folder+"angles.txt") as angle_file:
            for line in angle_file:
                line_values=line.split()
                image = cv2.imread(folder+line_values[0])
                resize_image = cv2.resize(image, (200,66))
                data_list.append([ToTensor()(resize_image.transpose(2,0,1)),float(line_values[1]) * np.pi / 180])
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, angle = self.data[index]
        return image, angle
