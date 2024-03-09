import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.loss import bbox2dist
from utils.metrics import bbox_iou, wasserstein_loss
from utils.loss import FocalLoss_YOLO, VarifocalLoss_YOLO, QualityfocalLoss_YOLO, EMASlideLoss, SlideLoss,RepLoss
from .atss import ATSSAssigner, generate_anchors
from utils.torch_utils import de_parallel
class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.iou_ratio = 0.5
        self.nwd_loss = False
        self.lrep_loss = False
        self.rep_loss = RepLoss(alpha=0.5, beta=0.5, sigma=0.5)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        
        if type(iou) is tuple:
            #wiou
            if len(iou) == 2:
                loss_iou = (iou[1].detach() * (1 - iou[0])).sum() / target_scores_sum
                iou = iou[0]
            else:
                loss_iou = (iou[0] * iou[1]).sum() / target_scores_sum
                iou = iou[2]
        else:
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum  # iou loss
            
        if self.nwd_loss:
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            nwd_loss = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = self.iou_ratio * loss_iou +  (1 - self.iou_ratio) * nwd_loss
            
        if self.lrep_loss:
            lrep=torch.tensor(0.0).to(pred_bboxes.device)
            bs=0
            for _,(mask,gt_batch_boxes, prd_batch_boxes) in enumerate(zip(fg_mask,pred_bboxes,  target_bboxes)):
                    if len(gt_batch_boxes[mask]) and len(prd_batch_boxes[mask]):
                        lrep += self.rep_loss(gt_batch_boxes[mask], prd_batch_boxes[mask])
                        bs+=1
            if bs>0:
                lrep = lrep / bs
        
            loss_iou +=lrep*0.01
        
                  
        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
        
        
class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        #h = model.args  # hyperparameters

        m = de_parallel(model).model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # self.bce = EMASlideLoss(nn.BCEWithLogitsLoss(reduction='none'))  # Exponential Moving Average Slide Loss
        # self.bce = SlideLoss(nn.BCEWithLogitsLoss(reduction='none')) # Slide Loss
        # self.bce = FocalLoss_YOLO(alpha=0.25, gamma=1.5) # FocalLoss
        # self.bce = VarifocalLoss_YOLO(alpha=0.75, gamma=2.0) # VarifocalLoss
        # self.bce = QualityfocalLoss_YOLO(beta=2.0) # QualityfocalLoss
        #self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        #self.assigner = ATSSAssigner(9, num_classes=self.nc)
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = batch
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        if isinstance(self.assigner, ATSSAssigner):
            anchors, _, n_anchors_list, _ = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
            target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(anchors, n_anchors_list, gt_labels, gt_bboxes, mask_gt, pred_bboxes.detach() * stride_tensor)
        # TAL
        else:
            target_labels, target_bboxes, target_scores, fg_mask,target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        if isinstance(self.bce, (nn.BCEWithLogitsLoss, FocalLoss_YOLO)):
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        elif isinstance(self.bce, VarifocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                cls_iou_targets.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * cls_iou_targets
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype)).sum() / max(fg_mask.sum(), 1)  # BCE
        elif isinstance(self.bce, QualityfocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                targets_onehot = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * targets_onehot
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                targets_onehot_pos = torch.where(fg_scores_mask > 0, targets_onehot, 0)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot_pos = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype), targets_onehot_pos.to(torch.bool)).sum() / max(fg_mask.sum(), 1)

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)

        if isinstance(self.bce, (EMASlideLoss, SlideLoss)):
            if fg_mask.sum():
                auto_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).mean()
            else:
                auto_iou = -1
            loss[1] = self.bce(pred_scores, target_scores.to(dtype), auto_iou).sum() / target_scores_sum  # BCE
        

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)