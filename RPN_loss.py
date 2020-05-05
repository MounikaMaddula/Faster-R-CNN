"""RPN Loss"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#custom imports
from rpn_network import RPN_Network
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RPN_Loss(nn.Module):

    def __init__(self, pos_thres, neg_thres, pos_ratio, n_samples,img_h, img_w, stride) :
        super(RPN_Loss,self).__init__()

        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        self.n_samples = n_samples
        self.pos_ratio = pos_ratio
        self.img_h = img_h
        self.img_w = img_w
        self.stride = stride

        #self.rpn_network = RPN_Network(in_channels,n_anchors,mode, pre_nms_train, post_nms_train, pre_nms_test,  \
        #            post_nms_test, nms_thres, img_h, img_w, stride, min_size)
        self.rpn_network = RPN_Network(512,n_anchors = 2,anc_ratios = [1,2], anc_scales = [1], mode = 'TRAIN', pre_nms_train = 12000, post_nms_train = 2000, pre_nms_test = 600,  \
                    post_nms_test = 300, nms_thres = 0.7, img_h = 2, img_w = 2, stride = 16, min_size = -1e-5)
        self.centre_anchors = self.rpn_network._generate_anchors()
        self.corner_anchors = self.rpn_network.corner_anchors

        self.reg_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def _calculate_offsets(self,targets, anchors, valid_index):

        anchors = anchors[valid_index.data]

        tx = (targets[:,1] - anchors[:,1])/anchors[:,3]
        ty = (targets[:,0] - anchors[:,0])/anchors[:,2]
        th = torch.log(targets[:,2]/anchors[:,2])
        tw = torch.log(targets[:,3]/anchors[:,3])

        return torch.cat([ty.unsqueeze(1),tx.unsqueeze(1),th.unsqueeze(1),tw.unsqueeze(1)],1)

    def _generate_targets(self,target_bnds):

        corner_targets = cxcy_to_corners(target_bnds)
        #corner_targets = target_bnds
        #Identifying valid anchor boxes by removing boxes falling
        corner_anchors = self.corner_anchors
        valid_index = torch.nonzero((corner_anchors[:,0]>=0) & (corner_anchors[:,1]>=0) & (corner_anchors[:,2]<=(self.img_h*self.stride)) &   \
                               (corner_anchors[:,3]<=(self.img_w*self.stride))).squeeze(1)
        corner_anchors = corner_anchors[valid_index,:]

        #IoU area of anchors & targets
        IoUs = IoU_area(corner_anchors,corner_targets, corners = True) #N,m

        #gt anchor boxes
        gt_overlap,gt_ind = IoUs.max(dim = 0) #1,m
        gt_ind = gt_ind.squeeze(0) #m
        #ground truth bounding box for each anchor box
        anchor_gt_overlap,anchor_gt_ind = IoUs.max(dim = 1) #N,

        labels = torch.ones(len(valid_index)).to(device)*-1
        #negative labels for anchors whose overlap is less than threshold
        try :
            labels[(anchor_gt_overlap < self.neg_thres).nonzero().squeeze(1).data] = 0
        except Exception as e :
            pass
        #positive labels to anchors whose overlap is more than threshold
        try :
            labels[(anchor_gt_overlap > self.pos_thres).nonzero().squeeze(1).data] = 1
        except Exception as e :
            pass
        #positive labels to groundtruth anchors
        labels[gt_ind.data] = 1

        #no of pos required
        n_pos = self.pos_ratio * self.n_samples
        #no of actual positives
        n_pos_actual = (labels==1).sum()
        if n_pos_actual > n_pos :
            #select random indexes of pos to ignore
            random_pos_index = (labels==1).nonzero().squeeze(1)[torch.randperm(int(n_pos_actual-n_pos))].squeeze(1)
            labels[random_pos_index] = -1

        n_neg_actual = (labels==0).sum()
        n_neg_req = self.n_samples - (labels==1).sum()
        if n_neg_actual > n_neg_req :
            #select random indexes of negs to ignore
            random_neg_index = (labels==0).nonzero()[torch.randperm(n_neg_actual-n_neg_req)].squeeze(1)
            labels[random_neg_index] = -1

        anchor_targets = target_bnds[anchor_gt_ind]
        anchor_targets = self._calculate_offsets(anchor_targets, self.centre_anchors, valid_index)
        return anchor_targets, labels, valid_index

    def forward(self,rpn_pred_bnd, rpn_pred_class, target_bnds):

        anchor_targets, labels, valid_index = self._generate_targets(target_bnds)

        rpn_pred_bnd = rpn_pred_bnd.squeeze(0)[valid_index]
        rpn_pred_class = rpn_pred_class.squeeze(0)[valid_index]

        obj_mask = (labels==1).nonzero().squeeze(1)
        non_obj_mask = (labels==0).nonzero().squeeze(1)

        pos_pred_bnd = rpn_pred_bnd[obj_mask,:]
        pos_target_bnd = anchor_targets[obj_mask,:]

        pos_pred_class = rpn_pred_class[obj_mask,:]
        neg_pred_class = rpn_pred_class[non_obj_mask,:]
        labels = Variable(labels).type(torch.LongTensor).to(device)
        #print (labels)

        loss_x = self.reg_loss(pos_pred_bnd[:,1], pos_target_bnd[:,1])
        loss_y = self.reg_loss(pos_pred_bnd[:,0], pos_target_bnd[:,0])
        loss_h = self.reg_loss(pos_pred_bnd[:,2], pos_target_bnd[:,2])
        loss_w = self.reg_loss(pos_pred_bnd[:,3], pos_target_bnd[:,3])

        loss_pos_class = self.ce_loss(pos_pred_class,labels[obj_mask])
        loss_neg_class = self.ce_loss(neg_pred_class, labels[non_obj_mask])

        return loss_x,loss_y, loss_h, loss_w, loss_pos_class, loss_neg_class
