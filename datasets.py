"""Code for dataloading"""

import pandas as pd
import torch
import os
from os.path import basename
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms.functional as FT
import xml.etree.ElementTree as ET
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from utils import corners_to_cxcy, cxcy_to_corners
from data_augment import transform

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Data_Loader(Dataset):

    def __init__(self, img_path,xml_path,split,out_size = (416,416)):

        super(Data_Loader,self).__init__()

        self.img_path = img_path
        self.xml_path = xml_path
        self.out_size = out_size
        self.split = split
        self.transforms = transforms.Compose([transforms.Resize((800,800)),transforms.ToTensor(),  \
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self._lables_dict()

        self.images = glob.glob(self.img_path+'/*.jpg')

    def _lables_dict(self):

        lables_dict = {}
        lables_dict['table'] = 1
        lables_dict['background'] = 0

        self.lables_dict = lables_dict

    def _resize(self,img, img_bboxes, scale):

        image = Image.open(img).convert('RGB')
        image = self.transforms(image) #c,h,w

        #img_bboxes - cx,cy,w,h
        img_bboxes[:,0] = img_bboxes[:,0] * scale[1]
        img_bboxes[:,1] = img_bboxes[:,1]*scale[0]
        img_bboxes[:,2] = img_bboxes[:,2] * scale[1]
        img_bboxes[:,3] =  img_bboxes[:,3]*scale[0]

        return image, img_bboxes


    def _parse_xml(self,filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        filename = root.find('filename').text

        for size in root.findall('size') :
            img_width, img_height,img_channel = size.find('width').text, size.find('height').text, size.find('depth').text

        lables = []
        bnd_boxes = []
        for obj in root.findall('object') :
            lables.append(obj.find('name').text.lower())
            for bnd in obj.findall('bndbox'):
                bnd_boxes.append([float(bnd.find('ymin').text),float(bnd.find('xmin').text),  \
                    float(bnd.find('ymax').text),float(bnd.find('xmax').text)])

        return img_width, img_height, img_channel, lables, bnd_boxes


    def __getitem__(self,ix):

        image = self.images[ix]
        #print (image)

        xml_file = self.xml_path+'/' + basename(image).replace('.jpg','.xml')

        img_width, img_height, img_channel, lables, bnd_boxes = self._parse_xml(xml_file)

        scale = 1/int(img_height),1/int(img_width)

        bnd_boxes = torch.from_numpy(np.asarray(bnd_boxes)) #M,xmin,ymin,xmax,ymax
        bnd_boxes = bnd_boxes.type(torch.FloatTensor)
        bnd_boxes = corners_to_cxcy(bnd_boxes) #M,cx,cy,w,h

        lables = [self.lables_dict[x] for x in lables]
        lables = torch.from_numpy(np.asarray(lables)) #M

        image = Image.open(image, mode='r')
        image = image.convert('RGB')

        image, bnd_boxes, lables  = transform(image, bnd_boxes, lables, split=self.split)
        #image,bnd_boxes = self._resize(image, bnd_boxes,scale = (1,1)) #3,h,w;M,4

        return image,bnd_boxes, lables

    def __len__(self):
        return len(self.images)

def main():

    check = Data_Loader(img_path = '../../YOLO/Data/Images',xml_path = '../../YOLO/Data/XMLs',split = 'TRAIN')

    dataloader =  DataLoader(check,batch_size=1,shuffle=True, num_workers=4)
    #c = 0
    for i in dataloader:
        try :
            image,bnd_boxes, lables =  i
            print (image.shape)
            print (bnd_boxes)
            print (lables)
        except Exception as e :
            print (e)
            print ('##############################################')
            exit()


if __name__ == '__main__':
    main()
