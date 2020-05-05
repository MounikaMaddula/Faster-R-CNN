"""Code for model training"""

from torch.autograd import Variable
import numpy as np
import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_train(model, train_dataloader,val_dataloader, rpn_loss, rcnn_loss,   \
        start_epoch,epochs, optimizer, exp_lr_decay, out_dir):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for param in model.parameters():
        param.requies_grad = True

    model = model.train()

    best_val_loss = np.inf
    #try :
    for epoch in range(start_epoch,epochs):
        net_rpn_loss = 0
        net_rpn_lossx = 0
        net_rpn_lossy = 0
        net_rpn_lossw = 0
        net_rpn_lossh = 0
        net_rpn_pos_loss = 0
        net_rpn_neg_loss = 0

        net_rcnn_loss = 0
        net_rcnn_lossx = 0
        net_rcnn_lossy = 0
        net_rcnn_lossw = 0
        net_rcnn_lossh = 0
        net_rcnn_class_loss = 0

        for images,gt_boxes,gt_labels in train_dataloader :
            images = Variable(images).to(device)
            gt_boxes = Variable(gt_boxes).squeeze(0).to(device)
            gt_labels = Variable(gt_labels).squeeze(0).to(device)

            #print (images.shape)

            rcnn_boxes, rcnn_class, rpn_pred_class, rpn_pred_bnd, proposal_rois = model(images)
            rpnloss_x,rpnloss_y, rpnloss_h, rpnloss_w,   \
                    rpnloss_pos_class, rpnloss_neg_class = rpn_loss(rpn_pred_bnd, rpn_pred_class, gt_boxes)
            rcnnloss_x, rcnnloss_y, rcnnloss_h, rcnnloss_w,  \
                    rcnnloss_class = rcnn_loss(proposal_rois,rcnn_boxes, rcnn_class,gt_boxes,gt_labels)

            rpn_box_loss = rpnloss_x + rpnloss_y + rpnloss_h + rpnloss_w
            rpn_class_loss = rpnloss_pos_class + rpnloss_neg_class
            rpnloss = rpn_box_loss + rpn_class_loss

            net_rpn_loss += rpnloss.item()
            net_rpn_lossx += rpnloss_x.item()
            net_rpn_lossy += rpnloss_y.item()
            net_rpn_lossw += rpnloss_w.item()
            net_rpn_lossh += rpnloss_h.item()
            net_rpn_pos_loss += rpnloss_pos_class.item()
            net_rpn_neg_loss += rpnloss_neg_class.item()

            rcnn_box_loss = rcnnloss_x + rcnnloss_y + rcnnloss_h + rcnnloss_w
            rcnnloss = rcnn_box_loss + rcnnloss_class

            net_rcnn_lossx += rcnnloss_x.item()
            net_rcnn_lossy += rcnnloss_y.item()
            net_rcnn_lossw += rcnnloss_w.item()
            net_rcnn_lossh += rcnnloss_h.item()
            net_rcnn_class_loss += rcnnloss_class.item()
            net_rcnn_loss += rcnnloss.item()

            loss = rpnloss + rcnnloss
            loss.backward()
            optimizer.step()
        exp_lr_decay.step()

        torch.save(model.state_dict(),'{0}/epoch-{1}.pth'.format(out_dir,epoch))

        print ('Epoch -------->',epoch)
        print ('RPN Training Loss ------------->',net_rpn_loss/len(train_dataloader))
        print ('RPN Training Loss X ------------->',net_rpn_lossx/len(train_dataloader))
        print ('RPN Training Loss Y ------------->',net_rpn_lossy/len(train_dataloader))
        print ('RPN Training Loss W ------------->',net_rpn_lossw/len(train_dataloader))
        print ('RPN Training Loss H ------------->',net_rpn_lossh/len(train_dataloader))
        print ('RPN Training Loss Pos Class ------------->',net_rpn_pos_loss/len(train_dataloader))
        print ('RPN Training Loss Neg Class ------------->',net_rpn_neg_loss/len(train_dataloader))
        print ('#'*50)

        print ('RCNN Training Loss ------------------>',net_rcnn_loss/len(train_dataloader))
        print ('RCNN Training Loss X ------------------>',net_rcnn_lossx/len(train_dataloader))
        print ('RCNN Training Loss Y ------------------>',net_rcnn_lossy/len(train_dataloader))
        print ('RCNN Training Loss W ------------------>',net_rcnn_lossw/len(train_dataloader))
        print ('RCNN Training Loss H ------------------>',net_rcnn_lossh/len(train_dataloader))
        print ('RCNN Training Loss Class ------------------>',net_rcnn_class_loss/len(train_dataloader))
        print ('#'*50)


        if epoch/5 == 0 :
            net_val_loss = model_eval(model, val_dataloader, rpn_loss, rcnn_loss, epoch)
            if net_val_loss < best_val_loss :
                torch.save(model.state_dict(),'{0}/best_val_epoch-{1}.pth'.format(out_dir,epoch))
                best_val_loss = net_val_loss
                print ('Best Val Loss -------------------->',best_val_loss)

    #except Exception as e :
        #print (rpn_pred)
        #print (rpn_pred_class, rpn_pred_bnd)
    #    print (rpn_pred_class, rpn_pred_bnd)
    #    print (proposal_rois.shape)


def model_eval(model, val_dataloader, rpn_loss, rcnn_loss,epoch):

    for param in model.parameters():
        param.requies_grad = False

    model = model.eval()

    net_rpn_loss = 0
    net_rpn_lossx = 0
    net_rpn_lossy = 0
    net_rpn_lossw = 0
    net_rpn_lossh = 0
    net_rpn_pos_loss = 0
    net_rpn_neg_loss = 0

    net_rcnn_loss = 0
    net_rcnn_lossx = 0
    net_rcnn_lossy = 0
    net_rcnn_lossw = 0
    net_rcnn_lossh = 0
    net_rcnn_class_loss = 0

    for images,gt_boxes,gt_labels in val_dataloader :
        images = Variable(images).to(device)
        gt_boxes = Variable(gt_boxes).squeeze(0).to(device)
        gt_labels = Variable(gt_labels).squeeze(0).to(device)

        rcnn_boxes, rcnn_class, rpn_pred_class, rpn_pred_bnd, proposal_rois = model(images)
        rpnloss_x,rpnloss_y, rpnloss_h, rpnloss_w,   \
                rpnloss_pos_class, rpnloss_neg_class = rpn_loss(rpn_pred_bnd, rpn_pred_class, gt_boxes)
        rcnnloss_x, rcnnloss_y, rcnnloss_h, rcnnloss_w,  \
                rcnnloss_class = rcnn_loss(proposal_rois,rcnn_boxes, rcnn_class,gt_boxes,gt_labels)

        rpn_box_loss = rpnloss_x + rpnloss_y + rpnloss_h + rpnloss_w
        rpn_class_loss = rpnloss_pos_class + rpnloss_neg_class
        rpnloss = rpn_box_loss + rpn_class_loss

        net_rpn_loss += rpnloss.item()
        net_rpn_lossx += rpnloss_x.item()
        net_rpn_lossy += rpnloss_y.item()
        net_rpn_lossw += rpnloss_w.item()
        net_rpn_lossh += rpnloss_h.item()
        net_rpn_pos_loss += rpnloss_pos_class.item()
        net_rpn_neg_loss += rpnloss_neg_class.item()

        rcnn_box_loss = rcnnloss_x + rcnnloss_y + rcnnloss_h + rcnnloss_w
        rcnnloss = rcnn_box_loss + rcnnloss_class

        net_rcnn_lossx += rcnnloss_x.item()
        net_rcnn_lossy += rcnnloss_y.item()
        net_rcnn_lossw += rcnnloss_w.item()
        net_rcnn_lossh += rcnnloss_h.item()
        net_rcnn_class_loss += rcnnloss_class.item()
        net_rcnn_loss += rcnnloss.item()

    print ('Epoch -------->',epoch)
    print ('RPN Validation Loss ------------->',net_rpn_loss/len(val_dataloader))
    print ('RPN Validation Loss X ------------->',net_rpn_lossx/len(val_dataloader))
    print ('RPN Validation Loss Y ------------->',net_rpn_lossy/len(val_dataloader))
    print ('RPN Validation Loss W ------------->',net_rpn_lossw/len(val_dataloader))
    print ('RPN Validation Loss H ------------->',net_rpn_lossh/len(val_dataloader))
    print ('RPN Validation Loss Pos Class ------------->',net_rpn_pos_loss/len(val_dataloader))
    print ('RPN Validation Loss Neg Class ------------->',net_rpn_neg_loss/len(val_dataloader))
    print ('#'*50)

    print ('RCNN Validation Loss ------------------>',net_rcnn_loss/len(val_dataloader))
    print ('RCNN Validation Loss X ------------------>',net_rcnn_lossx/len(val_dataloader))
    print ('RCNN Validation Loss Y ------------------>',net_rcnn_lossy/len(val_dataloader))
    print ('RCNN Validation Loss W ------------------>',net_rcnn_lossw/len(val_dataloader))
    print ('RCNN Validation Loss H ------------------>',net_rcnn_lossh/len(val_dataloader))
    print ('RCNN Validation Loss Class ------------------>',net_rcnn_class_loss/len(val_dataloader))
    print ('#'*50)

    return net_rpn_loss + net_rcnn_loss
