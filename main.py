
import pandas as pd
import numpy
import os
import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

#importing custom modules
from datasets import Data_Loader
from base_net import VGG16
from rpn_network import RPN_Network
from RPN_loss import RPN_Loss
from faster_rcnn import Faster_RCNN
from faster_rcnn_loss import Faster_RCNN_Loss
from train import model_train, model_eval

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():

    train_dataset = Data_Loader(img_path = '../../YOLO/Data/train_images',xml_path = '../../YOLO/Data/XMLs',split = 'TRAIN')
    val_dataset = Data_Loader(img_path = '../../YOLO/Data/test_images',xml_path = '../../YOLO/Data/XMLs',split = 'TEST')

    train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = False, num_workers = 4)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers = 4)

    model = Faster_RCNN(in_channels = 3,out_classes = 2,n_anchors = 15,anc_ratios = [1,2,3,0.5,0.33], anc_scales = [8,16,32],  \
                    mode = 'TRAIN', pre_nms_train = 12000, post_nms_train = 2000, pre_nms_test = 6000,  \
                post_nms_test = 300, nms_thres = 0.7, img_h = 50, img_w = 50, stride = 16, min_size = 8)

    rpn_loss = RPN_Loss(pos_thres = 0.5, neg_thres = 0.3, pos_ratio = 0.5, n_samples = 256,  \
                img_h = 50, img_w = 50, stride = 16)

    rcnn_loss = Faster_RCNN_Loss(n_sample = 128, pos_ratio = 0.25, pos_iou_thres = 0.5,   \
                neg_iou_thres_h = 0.3, neg_iou_thres_l = 0.0)

    optimizer = optim.Adam(model.parameters(), lr= 1e-6,betas=(0.5, 0.999))
    #optimizer = optim.RMSprop(model.parameters(), lr=0.00005)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 20, gamma=0.95)

    model.to(device)
    rpn_loss.to(device)
    rcnn_loss.to(device)

    model_train(model = model, train_dataloader = train_dataloader,val_dataloader = val_dataloader,  \
                rpn_loss = rpn_loss, rcnn_loss = rcnn_loss, start_epoch = 0,epochs = 100000, optimizer = optimizer,   \
                exp_lr_decay = exp_lr_scheduler, out_dir = 'chkpts')

if __name__ == '__main__':
    main()
