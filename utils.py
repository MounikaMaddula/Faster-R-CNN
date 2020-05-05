import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cxcy_to_corners(boxes):
    boxes = torch.cat([boxes[:,:2]-boxes[:,2:]/2, boxes[:,:2]+boxes[:,2:]/2],1)
    return boxes

def corners_to_cxcy(boxes):
    boxes = torch.cat([boxes[:,:2] + (boxes[:,2:] - boxes[:,:2])/2, boxes[:,2:] - boxes[:,:2]],1)
    return boxes

def intersection_area(boxes1, boxes2):

    xy_mins = torch.max(boxes1[:,:2].unsqueeze(1), boxes2[:,:2].unsqueeze(0)) # (n1, n2, 2)
    xy_maxs = torch.min(boxes1[:,2:].unsqueeze(1),boxes2[:,2:].unsqueeze(0))  # (n1, n2, 2)

    hw = torch.clamp(xy_maxs - xy_mins,min = 0)  # (n1, n2, 2)

    return hw[:,:,0]*hw[:,:,1]  # n1,n2

def area(box):
    return (box[:,3]-box[:,1])*(box[:,2]-box[:,0])

def IoU_area(boxes1,boxes2, corners = True):

    if not corners :
        boxes1 = cxcy_to_corners(boxes1)
        boxes2 = cxcy_to_corners(boxes2)

    area1 = area(boxes1) #n1
    area2 = area(boxes2) #n2
    intersection = intersection_area(boxes1, boxes2) #n1,n2
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection/union

def non_maximum_supression(boxes, scores,keep,iou_thres, corners = True):

    sort_val, sort_ind = torch.sort(scores, dim = 0, descending=True)

    if not corners :
        boxes = cxcy_to_corners(boxes)

    boxes = boxes[sort_ind,:]

    overlap = IoU_area(boxes,boxes).data

    suppress = torch.zeros(len(boxes)).to(device)
    # Consider each box in order of decreasing scores
    for box in range(boxes.shape[0]):
        # If this box is already marked for suppression
        if suppress[box] == 1:
            continue
        # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
        # Find such boxes and update suppress indices
        #print (suppress)
        #print (device)
        #print ((overlap[box] > iou_thres).type(torch.FloatTensor))
        suppress = torch.max(suppress, (overlap[box] > iou_thres).type(torch.FloatTensor).to(device))
        # The max operation retains previously suppressed boxes, like an 'OR' operation
        # Don't suppress this box, even though it has an overlap of 1 with itself
        suppress[box] = 0

    boxes = boxes[(suppress==0).nonzero().squeeze(1)]
    #boxes = boxes[1 - suppress]
    #print (boxes.shape)

    return boxes[:keep]
