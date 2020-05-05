"""Faster RCNN Loss"""

import torch
import torch.nn as nn

#custom imports
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Faster_RCNN_Loss(nn.Module):

    def __init__(self,n_sample, pos_ratio, pos_iou_thres, neg_iou_thres_h, neg_iou_thres_l):
        super(Faster_RCNN_Loss,self).__init__()

        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thres = pos_iou_thres
        self.neg_iou_thres_h = neg_iou_thres_h
        self.neg_iou_thres_l = neg_iou_thres_l

        self.reg_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def _generate_targets(self,gt_bnds,gt_labels,rois):

        #corner_rois = cxcy_to_corners(rois)
        corner_rois = rois
        centre_rois = corners_to_cxcy(rois)
        corner_gts = cxcy_to_corners(gt_bnds)
        #calculate IoU of rois w.r.t each gt_box
        IoUs = IoU_area(corner_rois, corner_gts)
        #anchor box for each gt object
        gt_overlap,gt_ind = IoUs.max(dim = 0) #1,m
        gt_ind = gt_ind.squeeze(0) #m
        #ground truth bounding box for each anchor box
        anchor_gt_overlap,anchor_gt_ind = IoUs.max(dim = 1) #N,

        labels = torch.ones(rois.shape[0]).to(device)*-1
        #labels = labels
        #negative labels for anchors whose overlap is less than threshold
        try :
            labels[((anchor_gt_overlap <= self.neg_thres_h) &   \
                    (anchor_gt_overlap > self.neg_thres_l)).nonzero().squeeze(1).data] = 0
        except :
            pass
        #positive labels to anchors whose overlap is more than threshold
        try :
            labels[(anchor_gt_overlap > self.pos_thres).nonzero().squeeze(1).data] = 1
        except :
            pass
        #positive labels to groundtruth anchors
        labels[gt_ind] = 1

        #no of pos required
        n_pos = self.pos_ratio * self.n_sample
        #no of actual positives
        n_pos_actual = (labels==1).sum()
        if n_pos_actual > n_pos :
            #select random indexes of pos to ignore
            random_pos_index = (labels==1).nonzero()[torch.randperm(int(n_pos_actual-n_pos))].squeeze(1)
            labels[random_pos_index] = -1

        n_neg_actual = (labels==0).sum()
        n_neg_req = self.n_sample - (labels==1).sum()
        if n_neg_actual > n_neg_req :
            #select random indexes of negs to ignore
            random_neg_index = (labels==0).nonzero()[torch.randperm(n_neg_actual-n_neg_req)].squeeze(1)
            labels[random_neg_index] = -1

        roi_targets = gt_bnds[anchor_gt_ind]
        roi_targets = self._calculate_offsets(roi_targets, centre_rois)
        roi_labels = gt_labels[anchor_gt_ind]

        pos_index = (labels==1).nonzero().squeeze(1)
        neg_index = (labels==0).nonzero().squeeze(1)
        keep_index = torch.cat((pos_index,neg_index),0)

        roi_targets = roi_targets[keep_index,:]
        roi_labels = roi_labels[keep_index]
        roi_labels[len(pos_index):] = 0

        return roi_targets, roi_labels, keep_index

    def _calculate_offsets(self,targets, rois):
        tx = (targets[:,1] - rois[:,1])/rois[:,3]
        ty = (targets[:,0] - rois[:,0])/rois[:,2]
        th = torch.log(targets[:,2]/rois[:,2])
        tw = torch.log(targets[:,3]/rois[:,3])

        return torch.cat([ty.unsqueeze(1),tx.unsqueeze(1),th.unsqueeze(1),tw.unsqueeze(1)],1)

    def forward(self,proposal_rois,rcnn_boxes, rcnn_class,gt_boxes,gt_labels):

        roi_targets, roi_labels,keep_index = self._generate_targets(gt_boxes,gt_labels,proposal_rois)

        rcnn_boxes = rcnn_boxes.squeeze(0)
        #print (proposal_rois.shape)
        #print (rcnn_boxes.shape)
        #print (rcnn_boxes)
        #print (keep_index)
        #print (proposal_rois.shape)
        rcnn_boxes = rcnn_boxes[keep_index,:]
        rcnn_class = rcnn_class.squeeze(0)
        rcnn_class = rcnn_class[keep_index,:]

        loss_x = self.reg_loss(rcnn_boxes[:,1], roi_targets[:,1])
        loss_y = self.reg_loss(rcnn_boxes[:,0], roi_targets[:,0])
        loss_h = self.reg_loss(rcnn_boxes[:,2], roi_targets[:,2])
        loss_w = self.reg_loss(rcnn_boxes[:,3], roi_targets[:,3])

        loss_class = self.ce_loss(rcnn_class,roi_labels)

        return loss_x, loss_y, loss_h, loss_w, loss_class
