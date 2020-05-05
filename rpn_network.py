"""RPN network"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def conv_layer(in_channels,out,kernel,stride,padding,bias, n = 1):
    layers = nn.Sequential()
    in_channels = in_channels
    for i in range(n) :
        layers.add_module('conv_{}'.format(i),nn.Conv2d(in_channels, out, kernel_size = kernel,stride=stride, padding=padding, bias= bias))
        layers.add_module('bn_{}'.format(i),nn.BatchNorm2d(out,momentum=0.9, eps=1e-5))
        layers.add_module('relu_{}'.format(i),nn.LeakyReLU(0.1, inplace = True))
        in_channels = out
    return layers

class RPN_Network(nn.Module):

    def __init__(self,in_channels,n_anchors,anc_ratios, anc_scales, mode, pre_nms_train, post_nms_train, pre_nms_test,  \
                post_nms_test, nms_thres, img_h, img_w, stride, min_size):
        super(RPN_Network,self).__init__()

        self.in_channels = in_channels
        self.n_anchors = n_anchors
        self.anc_ratios = anc_ratios
        self.anc_scales = anc_scales
        self.mode = mode
        self.pre_nms_train = pre_nms_train
        self.post_nms_train = post_nms_train
        self.pre_nms_test = pre_nms_test
        self.post_nms_test = post_nms_test
        self.nms_thres = nms_thres
        self.img_h = img_h
        self.img_w = img_w
        self.stride = stride
        self.min_size = min_size

        self.centre_anchors = self._generate_anchors()
        self.corner_anchors = cxcy_to_corners(self.centre_anchors)
        self.Net()

    def Net(self):
        self.conv = conv_layer(in_channels = self.in_channels,out = 512,kernel = 3,stride = 1,padding = 1,bias = False, n = 1)
        self.rpn_bnd = nn.Conv2d(in_channels = 512,out_channels = self.n_anchors * 4,kernel_size = 1,stride = 1,padding = 0,bias = True)
        self.rpn_class = nn.Conv2d(in_channels = 512,out_channels = self.n_anchors * 2,kernel_size = 1,stride = 1,padding = 0,bias = True)

    def forward(self,x):
        conv1 = self.conv(x)#N,512,50,50
        rpn_bnds = self.rpn_bnd(conv1) #N,n_anchors*4,50,50
        rpn_bnds = rpn_bnds.permute(0,2,3,1).contiguous().view(1, -1, 4)

        rpn_class = self.rpn_class(conv1) #N,n_anchors*2,50,50
        rpn_class = rpn_class.permute(0,2,3,1).contiguous().view(1, -1, 2)
        rpn_class = F.softmax(rpn_class,dim = 2)

        obj_scores = rpn_class.view(1, self.img_h, self.img_w, self.n_anchors, 2)[:, :, :, :, 1].contiguous().view(1, -1)
        proposal_rois = self._Proposal_rois(rpn_bnds,obj_scores)

        return conv1, proposal_rois, rpn_bnds, rpn_class

    def _generate_anchors(self):
        ctr_x = torch.arange(0.5,self.img_w + 0.5, step = 1)
        ctr_y = torch.arange(0.5,self.img_h + 0.5, step = 1)

        prior_boxes = []
        for x in ctr_x :
            for y in ctr_y :
                for scale in self.anc_scales :
                    for ratio in self.anc_ratios :
                        prior_boxes.append([y, x, scale * np.sqrt(ratio), scale * (1/np.sqrt(ratio))])

        prior_boxes = Variable(torch.FloatTensor(prior_boxes)).to(device)
        prior_boxes = prior_boxes*self.stride
        return prior_boxes

    def _process_rois(self, rpn_bnds, obj_scores):
        #Converting offsets to actual bounding box coordinates
        rpn_bnds = rpn_bnds.squeeze(0)
        obj_scores = obj_scores.squeeze(0)
        rois_y = rpn_bnds[:,0]*self.centre_anchors[:,2] + self.centre_anchors[:,0]
        rois_x = rpn_bnds[:,1]*self.centre_anchors[:,3] + self.centre_anchors[:,1]
        rois_h = torch.exp(rpn_bnds[:,2])*self.centre_anchors[:,2]
        rois_w = torch.exp(rpn_bnds[:,3])*self.centre_anchors[:,3]
        rois = torch.cat((rois_y.unsqueeze(1),rois_x.unsqueeze(1), rois_h.unsqueeze(1),rois_w.unsqueeze(1)),1)
        ##print (rois.shape)
        #moving to corner format to clip the boxes
        rois = cxcy_to_corners(rois)
        rois = rois.clamp(min = 0, max = self.img_h*self.stride)
        #moving back to centre format to remove rois without min height & width
        rois = corners_to_cxcy(rois)
        #print ('Pre min size shape',rois.shape)
        #print (rois[:,2].min(),rois[:,3].min())
        valid_rois = torch.nonzero((rois[:,2]>=self.min_size) & (rois[:,3]>=self.min_size)).squeeze(1)

        #required rois after filtering
        rois = rois[valid_rois,:]
        #print ('Post min shize shape',rois.shape)
        #converting rois back to corners format for NMS
        rois = cxcy_to_corners(rois)
        obj_scores = obj_scores[valid_rois]

        return rois, obj_scores

    def _Proposal_rois(self,rpn_bnds, obj_scores):

        rois, obj_scores = self._process_rois(rpn_bnds, obj_scores)
        #selecting pre nms rois based on scores
        _,ind = obj_scores.sort(dim = 0, descending = True)
        ##print (obj_scores)
        obj_scores = obj_scores[ind]
        rois = rois[ind]

        if self.mode == 'TRAIN':
            obj_scores = obj_scores[:self.pre_nms_train]
            rois = rois[:self.pre_nms_train,:]
            ##print ('PRE NMS Shape',rois.shape)
            rois = non_maximum_supression(boxes = rois, scores = obj_scores,keep = self.post_nms_train,  \
                                            iou_thres = self.nms_thres)
            ##print ('post NMS shape',rois.shape)
            return rois[:self.post_nms_train]
        else :
            obj_scores = obj_scores[:self.pre_nms_test]
            rois = rois[:self.pre_nms_test,:]
            rois = non_maximum_supression(boxes = rois, scores = obj_scores,keep = self.post_nms_test,  \
                                            iou_thres = self.nms_thres)

            return rois[:self.post_nms_test]
