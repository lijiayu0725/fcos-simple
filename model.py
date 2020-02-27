import math
import torch
from torch import nn
from torch.nn import functional as F

from fpn import ResNet50FPN

from loss import FocalLoss, IoULoss
from utils import bias_init_with_prob, distance2bbox, to_onehot

INF = 1e8

class FCOS(nn.Module):
    def __init__(self, classes=80,
                 state_dict_path='/Users/nick/.cache/torch/checkpoints/resnet50-19c8e357.pth',
                 strides=(8, 16, 32, 64, 128),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF))):
        super(FCOS, self).__init__()
        self.backbone = ResNet50FPN(state_dict_path=state_dict_path)
        self.name = 'RetinaNet'
        self.classes = classes
        self.strides = strides
        self.regress_ranges = regress_ranges

        self.threshold = 0.05
        self.top_n = 1000
        self.nms = 0.5
        self.detections = 100

        def make_head():
            layers = []
            for _ in range(4):
                layers += [nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.GroupNorm(32, 256), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.cls_convs = make_head()
        self.reg_convs = make_head()

        self.fcos_cls = nn.Conv2d(256, 80, kernel_size=3, padding=1)
        self.fcos_reg = nn.Conv2d(256, 4, kernel_size=3, padding=1)
        self.fcos_centerness = nn.Conv2d(256, 1, kernel_size=3, padding=1)

        self.initialize()

        self.cls_criterion = FocalLoss()
        self.box_criterion = IoULoss()
        self.centerness_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def initialize(self):
        self.backbone.initialize()

        bias = bias_init_with_prob(0.01)

        def initialize_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, bias)

        self.cls_convs.apply(initialize_layer)
        self.reg_convs.apply(initialize_layer)

        self.fcos_cls.apply(initialize_layer)
        self.fcos_reg.apply(initialize_layer)
        self.fcos_centerness.apply(initialize_layer)

    def transform_targets(self, targets):
        targets[:, :, 4] = targets[:, :, 4] + 1
        targets[:, :, 2] += targets[:, :, 0]
        targets[:, :, 3] += targets[:, :, 1]
        return targets

    def forward(self, x):
        if self.training:
            x, targets = x
        targets = self.transform_targets(targets)
        features = []
        features.extend(self.backbone(x))

        cls_heads = [self.cls_convs(t) for t in features]
        reg_heads = [self.reg_convs(t) for t in features]

        cls_scores = [self.fcos_cls(t) for t in cls_heads]
        centerness_scores = [self.fcos_centerness(t) for t in cls_heads]
        reg_scores = [self.fcos_reg(t).float().exp() for t in reg_heads]

        if self.training:
            return self.loss(cls_scores, reg_scores, centerness_scores, targets.float())

    def loss(self, cls_scores, bbox_preds, centernesses, targets):
        featmap_sizes = [score.shape[-2:] for score in cls_scores]
        all_level_points = self.getpoints(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, bbox_targets = self.fcos_target(all_level_points, targets)

        num_imgs = cls_scores[0].shape[0]

        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, 80) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.cls_criterion(flatten_cls_scores, to_onehot(flatten_labels)).sum() / (num_imgs + num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)

            # centerness weighted iou loss
            loss_bbox = self.box_criterion(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets).sum() / pos_centerness_targets.sum()
            loss_centerness = self.centerness_criterion(pos_centerness, pos_centerness_targets).mean()
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
        return loss_cls, loss_bbox, loss_centerness

    def centerness_target(self, pos_bbox_targets):
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
            *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        )
        return centerness_targets


    def fcos_target(self, points, targets):
        num_levels = len(points)

        # [5] -> (num_points, 2)
        expanded_regress_ranges = [points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)]
        # (all_num_points, 2)
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        labels_list = []
        bbox_targets_list = []
        for i in range(len(targets)):
            valid = targets[i][:, 4] >= 0
            gt_bboxes = targets[i][valid, :4]
            gt_labels = targets[i][valid, 4]
            num_points = concat_points.shape[0]
            num_gts = gt_labels.shape[0]
            if num_gts == 0:
                labels_list.append(gt_labels.new_zeros(num_points))
                bbox_targets_list.append(gt_bboxes.new_zeros((num_points, 4)))
                continue
            areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)

            # (num_points, num_gts)
            areas = areas[None].repeat(num_points, 1)

            # (num_points, num_gts, 2)
            regress_ranges_tmp = concat_regress_ranges[:, None, :].expand(num_points, num_gts, 2)
            gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
            xs, ys = concat_points[:, 0], concat_points[:, 1]
            xs = xs[:, None].expand(num_points, num_gts)
            ys = ys[:, None].expand(num_points, num_gts)

            # calculate the distance for each points and gt_bboxes
            left = xs - gt_bboxes[..., 0]
            right = gt_bboxes[..., 2] - xs
            top = ys - gt_bboxes[..., 1]
            bottom = gt_bboxes[..., 3] - ys

            # (num_points, num_gts, 4)
            bbox_targets = torch.stack((left, top, right, bottom), dim=-1)

            inside_gt_bbox_mask = bbox_targets.min(dim=-1)[0] > 0
            max_regress_distance = bbox_targets.max(dim=-1)[0]
            inside_regress_range = (
                (max_regress_distance >= regress_ranges_tmp[..., 0])
                &
                (max_regress_distance <= regress_ranges_tmp[..., 1])
            )
            areas[inside_gt_bbox_mask == False] = INF
            areas[inside_regress_range == False] = INF
            min_area, min_area_inds = areas.min(dim=1) # (num_points, )

            labels = gt_labels[min_area_inds]
            labels[min_area == INF] = 0
            bbox_targets = bbox_targets[range(num_points), min_area_inds]
            labels_list.append(labels)
            bbox_targets_list.append(bbox_targets)
        num_points = [center.shape[0] for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets # [5] -> (num_points, ), [5] -> (num_points, 4)

    # get the center points of each featmap
    def getpoints(self, featmap_sizes, dtype, device):
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            h, w = featmap_sizes[i]
            x_range = torch.arange(0, w * self.strides[i], self.strides[i], dtype=dtype, device=device)
            y_range = torch.arange(0, h * self.strides[i], self.strides[i], dtype=dtype, device=device)
            y, x = torch.meshgrid([y_range, x_range])
            mlvl_points.append(torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + self.strides[i] // 2)
        return mlvl_points

    def fix_bn(self):
        def fix_batchnorm_param(layer):
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

        self.apply(fix_batchnorm_param)

    def train(self, mode=True):
        super(FCOS, self).train(mode)

        return self
