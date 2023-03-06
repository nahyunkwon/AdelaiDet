# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

from detectron2.config import configurable
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

class NeRF_Classifier_FCOS(nn.Module):
    def __init__(self, nerfWeights, flagConv4feats=False, actMapDim=256):
        # NeRFWeights: 
        #    C x d (C classes x feature dimension), 
        #    format: torch tensor in cpu
        # featTransform: whether to apply more conv layers to the input activation maps
        super(NeRF_Classifier_FCOS, self).__init__()
        self.C_classes = nerfWeights.shape[0]  # C classes
        self.nerfFeaDim = nerfWeights.shape[1]  # the dimension of vectorized NeRF weights, e.g., 10000
        self.head_param = nerfWeights[:, :, None, None]  # C x 10000 x 1 x 1

        self.flagConv4feats = flagConv4feats
        self.actMapDim = actMapDim  # e.g., 1024 based on FasterRCNN activation maps

        self.convLayer4nerfWeightsTransform = nn.Conv2d(self.nerfFeaDim, self.actMapDim, kernel_size=1)
        # self.convLayer4nerfWeightsTransform = nn.Sequential(OrderedDict([
        #         ('conv1', nn.Conv2d(self.nerfFeaDim, self.actMapDim, kernel_size=1)),
        #         ('relu1', nn.ReLU()),
        #     ]))

        # trasform from num classes -> num classes * num_anchors
        # self.classesToAnchors = nn.Conv2d(self.C_classes, self.C_classes, kernel_size=1)
        self.classesToclasses = nn.Conv2d(self.C_classes, self.C_classes, kernel_size=3, stride=1, padding=1)
        # self.classesToAnchors = nn.Sequential(OrderedDict([
        #         ('conv1', nn.Conv2d(self.C_classes, self.C_classes * num_anchors, kernel_size=1)),
        #         ('relu1', nn.ReLU()),
        #     ]))

        # whether to apply more conv layers to the input activation maps; need to be nonlinear otherwise not needed.
        if self.flagConv4feats:
            self.convLayer4feaTransform = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(self.actMapDim, self.actMapDim, 1)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(self.actMapDim, self.actMapDim, 1)),
                # ('tanh', nn.Tanh()),
            ]))

    def forward(self, featMap):
        # self.head_param: C x 10000 x 1 x 1  -->  C x 1800 x 1 x 1
        # print("===================================== head_param", self.head_param[0, :10, ...])
        self.classifier_weights = self.convLayer4nerfWeightsTransform(self.head_param)
        self.classifier_weights = self.classifier_weights.squeeze().squeeze()  # C x 256

        # print("======================= featMap", featMap.shape) [B_size, 12502, 256] -> [B_size, 12502, num_classes]
        # featMap: N x 1024
        feat_H, feat_W = featMap.shape[-2], featMap.shape[-1]
        self.featMap = featMap
        if self.flagConv4feats:
            self.featMap = self.convLayer4feaTransform(self.featMap)

        # print("======================= self.featMap", self.featMap.shape)
        self.featMap = torch.permute(self.featMap, (0, 2, 3, 1))  # N x H X W x C_in (256)
        self.classifier_weights = torch.permute(self.classifier_weights, (1, 0))  # 1024 x C

        out = torch.matmul(self.featMap, self.classifier_weights)  # N x H X W x C
        # print("out: ", out.min(), out.max())
        out = torch.permute(out, [0, 3, 1, 2])  # N x C x H x W
        out = self.classesToclasses(out)
        return out
