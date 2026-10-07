"""Two-class (background + foreground) Mask R-CNN built from torchvision COCO weights."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_mask_rcnn_binary(
    num_classes: int = 2,
    *,
    pretrained_backbone: bool = True,
) -> nn.Module:
    """
    ``num_classes`` includes background (COCO convention: e.g. 2 = bg + one cell line).

    If ``pretrained_backbone`` is True, load COCO Mask R-CNN weights and replace ROI heads
    for ``num_classes``. If False, ImageNet backbone + randomly initialized detection heads.
    """
    if pretrained_backbone:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        return model
    return maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
