import torch
import torch.nn.functional as F

from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh
        self.roi_heads.score_thresh = 0.25

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    def detect_with_proposal(self, img, t_1_proposal):
        """
            https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py
        """
        images = img
            
        device = list(self.parameters())[0].device
        images = images.to(device)
        
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, _ = self.transform(images, None)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        if not len(t_1_proposal):
            return torch.Tensor([]), torch.Tensor([])

        tt = resize_boxes(
                t_1_proposal.to(device),
                original_image_sizes[0], images.image_sizes[0]
        )

        # detections, _ = self.roi_heads(features, tt, images.image_sizes, None)
        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]

        box_features = self.roi_heads.box_roi_pool(
            features, [tt], images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(
            box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, [tt])
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(
            pred_boxes, images.image_sizes[0], original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()

        return pred_boxes, pred_scores
