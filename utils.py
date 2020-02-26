import math

import torch
from torch import nn


def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = torch.zeros((batch_size, ndetections), device=device)
    out_boxes = torch.zeros((batch_size, ndetections, 4), device=device)
    out_classes = torch.zeros((batch_size, ndetections), device=device)

    for batch in range(batch_size):
        keep = (all_scores[batch, :].view(-1) > 0).nonzero()
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, 4)
        classes = all_classes[batch, keep].view(-1)

        if scores.nelement() == 0:
            continue

        scores, indices = torch.sort(scores, descending=True)
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).view(-1)
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():  # ???
                i -= 1
                break
            # compute iou over boxes
            xy1 = torch.max(boxes[:, :2], boxes[i, :2])
            xy2 = torch.min(boxes[:, 2:], boxes[i, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), dim=1)
            criterion = ((scores > scores[i]) |
                         (inter / (areas + areas[i] - inter) <= nms) |
                         (classes != classes[i]))
            criterion[i] = 1  # criterion -> no useful any more

            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[criterion.nonzero(), :].view(-1, 4)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[batch, :i + 1] = scores[:i + 1]
        out_boxes[batch, :i + 1, :] = boxes[:i + 1, :]
        out_classes[batch, :i + 1] = classes[:i + 1]

    return out_scores, out_boxes, out_classes

class Scale(nn.Module):
    """
    A learnable scale parameter
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(math.log((1 - prior_prob) / prior_prob))
    return bias_init

def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)

def bbox_overlaps(bboxes1, bboxes2):

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, 1)

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    overlap = wh[:, 0] * wh[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)


    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap)

    return ious

def to_onehot(target, class_num=80):
    target = target.unsqueeze(-1).long()
    one_hot = torch.zeros(target.shape[0], class_num+1, device=target.device).scatter_(1, target, 1)
    one_hot = one_hot[:, 1:]
    return one_hot

if __name__ == '__main__':
    target = torch.tensor([0, 0, 2, 4])
    print(to_onehot(target))