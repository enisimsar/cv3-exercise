import torch
import torch.nn.functional as F

from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()
        return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

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

        # for bbox in pred_boxes:
        #     bbox[0] = max(bbox[0], 0)
        #     bbox[1] = max(bbox[1], 0)
        #     bbox[2] = min(bbox[2], original_image_sizes[0][0])
        #     bbox[3] = min(bbox[3], original_image_sizes[0][0])

        return pred_boxes, pred_scores

    def detect_with_proposals(self, img, t_1_proposals):
        """
            https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py
        """     
        with torch.no_grad():       
            images = img
            device = list(self.parameters())[0].device
            images = images.to(device)

            original_image_sizes = []
            for img in images:
                val = img.shape[-2:]
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))

            images, _ = self.transform(images, None)
            features = self.backbone(torch.cat([images.tensors for _ in range(len(t_1_proposals))], dim=0))
            
            tt = []
            for t in t_1_proposals:
                tt.append(
                    resize_boxes(
                        torch.Tensor(t).unsqueeze(0).to(device),
                        original_image_sizes[0], images.image_sizes[0]
                    )
                )

            detections, _ = self.roi_heads(features, tt, images.image_sizes * len(tt), None)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

            return [(detection['boxes'].detach().cpu(), detection['scores'].detach().cpu()) for detection in detections]

    def predict_boxes(self, images, boxes):
        device = list(self.parameters())[0].device
        images = images.to(device)
        boxes = boxes.to(device)

        targets = None
        original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        # proposals, proposal_losses = self.rpn(images, features, targets)
        from torchvision.models.detection.transform import resize_boxes
        boxes = resize_boxes(
            boxes, original_image_sizes[0], images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(
            box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)


        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(
            pred_boxes, images.image_sizes[0], original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores
        