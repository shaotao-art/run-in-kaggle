import torch
from torch import nn
from loss.ohem import ohem_batch
from loss.dice_loss import DiceLoss
from loss.emb_loss_v1 import EmbLoss_v1
from loss.iou import iou



class PANLoss(nn.Module):
    def __init__(self, text_loss, kernel_loss, emb_loss) -> None:
        super().__init__()
        self.text_loss = DiceLoss(**text_loss)
        self.kernel_loss = DiceLoss(**kernel_loss)
        self.emb_loss = EmbLoss_v1(**emb_loss)


    def forward(self, out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes):
                # output
        texts = out[:, 0, :, :]
        kernels = out[:, 1:2, :, :]
        embs = out[:, 2:, :, :]

        # text loss
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.text_loss(
            texts, gt_texts, selected_masks, reduce=False)
        iou_text = iou(
            (texts > 0).long(), gt_texts, training_masks, reduce=False)
        losses = dict(loss_text=loss_text, iou_text=iou_text)

        # kernel loss
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(
                kernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou(
            (kernels[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :],
            training_masks * gt_texts, reduce=False)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))

        # embedding loss
        loss_emb = self.emb_loss(
            embs, gt_instances, gt_kernels[:, -1, :, :], training_masks,
            gt_bboxes, reduce=False)
        losses.update(dict(loss_emb=loss_emb))

        return losses
    
