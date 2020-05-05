"""Faster R-CNN architecture"""

import torch
import torch.nn as nn
from torchvision.ops import roi_align
import torch.nn.functional as F

#custom imports
from base_net import VGG16
from rpn_network import RPN_Network
from RPN_loss import RPN_Loss
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Faster_RCNN(nn.Module):

    def __init__(self,in_channels,out_classes,n_anchors,anc_ratios, anc_scales,  \
                    mode, pre_nms_train, post_nms_train, pre_nms_test,  \
                post_nms_test, nms_thres, img_h, img_w, stride, min_size):
        super(Faster_RCNN,self).__init__()

        self.stride = stride
        self.out_classes = out_classes

        self.basenet = VGG16(in_channels = in_channels)
        self.rpn_network = RPN_Network(512,n_anchors,anc_ratios, anc_scales,mode, pre_nms_train, post_nms_train, pre_nms_test,  \
                    post_nms_test, nms_thres, img_h, img_w, stride, min_size)
        #self.roi_align = roi_align(output_size = (7,7), spatial_scale=1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((7,7))
        self.Net()

    def _process_rois(self,rois, conv1):
        """
        size = (7, 7)
        adaptive_max_pool = AdaptiveMaxPool2d(size[0], size[1])
        """
        #adding index to rois for pooling
        rois_inds = torch.zeros(rois.shape[0]).view(-1,1).to(device)
        #arranging rois for pooling
        #corner_rois = cxcy_to_corners(rois)
        rois = torch.cat((rois_inds,rois),dim = 1) #N,ind,y1,x1,y2,x2
        rois = rois[:,[0, 2, 1, 4, 3]].contiguous() #N,ind,x1,y1,x2,y2
        #print (rois.shape)
        #sampling the rois as per stride
        rois[:, 1:].mul_(1/self.stride)
        rois = rois.long()
        #output = roi_align(conv1,rois,(7,7))
        #output = self.roi_align(input = conv1, boxes = rois)

        output = []
        num_rois = rois.size(0)
        for i in range(num_rois):
            roi = rois[i]
            im_idx = roi[0]
            im = conv1.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
            output.append(self.adaptive_max_pool(im))
        output = torch.cat(output, 0).to(device)

        return output

    def Net(self):
        self.layer1 = nn.Linear(25088, 4096)
        self.layer2 = nn.Linear(4096,4096)
        self.rcnn_boxes = nn.Linear(4096,4)
        self.rcnn_class = nn.Linear(4096,self.out_classes)

    def forward(self,x):
        #print (x.shape)
        img_features = self.basenet(x)
        conv1, proposal_rois, rpn_bnd, rpn_class = self.rpn_network(img_features)
        out = self._process_rois(proposal_rois, img_features)
        #print (out.shape)
        out = out.view(out.shape[0],-1)
        out = F.leaky_relu(self.layer1(out),0.1)
        out = F.leaky_relu(self.layer2(out),0.1)
        rcnn_boxes = self.rcnn_boxes(out)
        rcnn_class = self.rcnn_class(out)
        rcnn_class = F.softmax(rcnn_class,dim = 1)
        return rcnn_boxes, rcnn_class, rpn_class, rpn_bnd, proposal_rois
