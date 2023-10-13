from micromind import MicroMind

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss, TaskAlignedAssigner
from micromind.utils.parse import parse_arguments

from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

import sys; sys.path.append("/home/franz/dev/micromind/yolo_teo")
from modules import YOLOv8

class Loss(v8DetectionLoss):
    def __init__(self, h, m, device):    # model must be de-paralleled
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

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
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        print(loss, [torch.std_mean(f) for f in feats])
        torchvision.utils.save_image(batch["img"], "samples.png")
        breakpoint()

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

class YOLO(MicroMind):
    def __init__(self, m_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        w, r, d = 1, 1, 1
        self.modules["yolo"] = YOLOv8(1, 1, 1, 80)

        self.modules["yolo"].load_state_dict(
            torch.load("/home/franz/dev/micromind/yolo_teo/yolov8l.pt"
        ))

        self.m_cfg = m_cfg

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        preprocessed_batch = {}
        preprocessed_batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and k != "img":
                preprocessed_batch[k] = batch[k].to(self.device)

        return preprocessed_batch

    def forward(self, batch):
        preprocessed_batch = self.preprocess_batch(batch)

        return self.modules["yolo"](preprocessed_batch["img"].to(self.device))

    def compute_loss(self, pred, batch):
        self.criterion = Loss(self.m_cfg, self.modules["yolo"].head, self.device)
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi =  self.criterion(
            pred[1],    # pass elements at the beginning of the backward graph
            preprocessed_batch
        )

        return lossi_sum

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.modules.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", factor=0.1, patience=25, threshold=5, verbose=True
        )
        return opt, sched


if __name__ == "__main__":
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.cfg  import get_cfg

    m_cfg = get_cfg("yolo_cfg/default.yaml")
    data_cfg = check_det_dataset("yolo_cfg/coco8.yaml")
    batch_size = 8

    # coco8_dataset = build_yolo_dataset(
        # m_cfg, mode="train", "/mnt/data/coco8", batch_size, data_cfg
    # )
    mode = "train"
    coco8_dataset = build_yolo_dataset(m_cfg, "/mnt/data/coco8", batch_size, data_cfg, mode=mode, rect=mode == "val")

    coco8_data = torch.utils.data.Subset(coco8_dataset, [1])

    loader = DataLoader(
        coco8_dataset, batch_size, collate_fn=getattr(coco8_dataset, 'collate_fn', None)
    )


    # def preprocess_batch(batch):
        # """Preprocesses a batch of images by scaling and converting to float."""
        # batch['img'] = batch['img'].to(batch["img"].device, non_blocking=True).float() / 255
        # for k in batch:
            # if isinstance(k, torch.Tensor):
                # print(k)
                # batch[k] = batch[k].to(self.device)
        # return batch
# 
    # for b in loader:
        # b = preprocess_batch(b)
        # torchvision.utils.save_image(b["img"], "batch1.png")
        # print(b["img"].shape, torch.std_mean(b["img"]))
# 
    # for b in loader:
        # b = preprocess_batch(b)
        # torchvision.utils.save_image(b["img"], "batch2.png")
        # print(b["img"].shape, torch.std_mean(b["img"]))

    hparams = parse_arguments()
    m = YOLO(
        m_cfg,
        hparams=hparams
    )

    m.train(
        epochs=300,
        datasets={"train": loader, "val": loader},
    )

    m.test(
        datasets={"test": testloader},
    )
