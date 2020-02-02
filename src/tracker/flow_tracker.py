import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class FlowTracker(nn.Module):
    def __init__(self):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FlowTracker, self).__init__()

        self.backbone = backbone

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0],
            output_size=7,
            sampling_ratio=2)

        # resolution = box_roi_pool.output_size[0]
        # representation_size = 1024
        # box_head = TwoMLPHead(
        #     self.backbone.out_channels * resolution ** 2,
        #     representation_size)

        # box_predictor = FastRCNNPredictor(
        #     representation_size,
        #     2
        # )

        # """
        #     box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        #     box_batch_size_per_image=512, box_positive_fraction=0.25,
        #     bbox_reg_weights=None,
        #     box_score_thresh=0.25, box_nms_thresh=0.6, box_detections_per_img=100)
        # """
        # self.roi_heads = RoIHeads(
        #     # Box
        #     box_roi_pool, box_head, box_predictor,
        #     0.5, 0.5,
        #     512, 0.25,
        #     None,
        #     0.25, 0.6, 100
        # )

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(800, 1333, image_mean, image_std)

        self.conv = nn.Conv2d(2, 3, kernel_size=1)
        self.fc = nn.Linear(256 * 7 * 7, 4)

    def forward(self, images, boxes):
        device = list(self.parameters())[0].device
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images = self.conv(images)
        images, _ = self.transform(images, None)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # print(features)

        tt = torch.cat([resize_boxes(
            bb.to(device),
            original_image_sizes[0], images.image_sizes[0]
        ) for bb in boxes], 0)

        box_features = self.box_roi_pool(
            features, [tt], images.image_sizes)

        box_features = box_features.view(-1, 256 * 7 * 7)

        # print(box_features.shape)

        pred_boxes = self.fc(box_features)
    
        return pred_boxes