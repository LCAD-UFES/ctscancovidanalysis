import os
import numpy as np
from torch.utils.data import Dataset
from pydicom import dcmread
import matplotlib.pyplot as plt
from nibabel.testing import data_path
import nibabel as nib
from PIL import Image

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, classes, covid_files, non_covid_files, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.files_path = [non_covid_files, covid_files]
        self.image_list = []

        # read the files from data split text files
        covid_files = read_txt(covid_files)
        non_covid_files = read_txt(non_covid_files)

        # combine the positive and negative files into a cummulative files list
        for cls_index in range(len(self.classes)):
            
            class_files = [[os.path.join(self.root_dir, x), cls_index] \
                            for x in read_txt(self.files_path[cls_index])]
            self.image_list += class_files
            
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx][0]
        label = int(self.image_list[idx][1])

        # Read nii.gz
        image = nib.load(path)
        #numpy array
        image = image.get_fdata()
       
        # print(image.shape)
        # print(path)
        # plt.imshow(image)
        # plt.show()

        # Apply transforms
        if self.transform:
            image = self.transform(image)


        data = {'img':   image,
                'label': label,
                'paths' : path}

        return data
