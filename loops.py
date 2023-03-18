import os
import torch
import cv2
from utils.utilers import AverageMeter, write_result_ctw
from post_processing import pa
import numpy as np
from tqdm import tqdm
import os
from utils.utilers import get_config



def train_step(model, batch, criterion, device):
    """
    should return a dict with key 'loss' that all the losses are gathered.
    all the other keys in the dict will be printed to the conslo
    """
    out = dict()
    losses = AverageMeter(max_len=500)
    losses_text = AverageMeter(max_len=500)
    losses_kernels = AverageMeter(max_len=500)
    losses_emb = AverageMeter(max_len=500)


    ious_text = AverageMeter(max_len=500)
    ious_kernel = AverageMeter(max_len=500)
    
    for k in batch.keys():
        batch[k] = batch[k].to(device)
    
    pred = model(batch['imgs'])
    outputs = criterion(pred, batch['gt_texts'], batch['gt_kernels'], batch['training_masks'], batch['gt_instances'], batch['gt_bboxes'])
    
    # detection loss
    loss_text = torch.mean(outputs['loss_text'])
    losses_text.update(loss_text.item(), batch['imgs'].size(0))
    out['loss_text'] = losses_text.avg

    loss_kernels = torch.mean(outputs['loss_kernels'])
    losses_kernels.update(loss_kernels.item(), batch['imgs'].size(0))
    out['loss_kernel'] = losses_kernels.avg

    loss_emb = torch.mean(outputs['loss_emb'])
    losses_emb.update(loss_emb.item(), batch['imgs'].size(0))
    out['loss_embed'] = losses_emb.avg

    loss = loss_text + loss_kernels + loss_emb
    losses.update(loss.item(), batch['imgs'].size(0))
    out['loss'] = loss

    iou_text = torch.mean(outputs['iou_text'])
    ious_text.update(iou_text.item(), batch['imgs'].size(0))
    iou_kernel = torch.mean(outputs['iou_kernel'])
    ious_kernel.update(iou_kernel.item(), batch['imgs'].size(0))
    out['ious_text'] = ious_text.avg
    out['ious_kernel'] = ious_kernel.avg
    return out


def valid(dataloader, model, device):
    config = get_config('./pan-config.yaml')
    with torch.no_grad():
        loop = tqdm(dataloader)
        for sample in loop:
            img = sample['imgs']
            out = model(img.to(device))
        
            outputs = dict()
            img_meta = sample['img_metas']

            score = torch.sigmoid(out[:, 0, :, :])
            kernels = out[:, :2, :, :] > 0
            text_mask = kernels[:, :1, :, :]
            kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
            emb = out[:, 2:, :, :]
            emb = emb * text_mask.float()

            score = score.data.cpu().numpy()[0].astype(np.float32)
            kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
            emb = emb.cpu().numpy()[0].astype(np.float32)

            # pa
            label = pa(kernels, emb)

            # image size
            org_img_size = img_meta['org_img_size'][0]
            img_size = img_meta['img_size'][0]
            img_path = img_meta['img_path'][0]
            img_name = img_meta['img_name'][0]

            label_num = np.max(label) + 1
            label = cv2.resize(label, (int(img_size[1]), int(img_size[0])),
                                interpolation=cv2.INTER_NEAREST)
            score = cv2.resize(score, (int(img_size[1]), int(img_size[0])),
                                interpolation=cv2.INTER_NEAREST)


            scale = (float(org_img_size[1]) / float(img_size[1]),
                        float(org_img_size[0]) / float(img_size[0]))

            bboxes = []
            scores = []
            for i in range(1, label_num):
                ind = label == i
                points = np.array(np.where(ind)).transpose((1, 0))

                if points.shape[0] < config['test_cfg']['min_area']:
                    label[ind] = 0
                    continue

                score_i = np.mean(score[ind])
                if score_i < config['test_cfg']['min_score']:
                    label[ind] = 0
                    continue


                if config['test_cfg']['bbox_type'] == 'rect':
                    rect = cv2.minAreaRect(points[:, ::-1])
                    bbox = cv2.boxPoints(rect) * scale
                elif config['test_cfg']['bbox_type']  == 'poly':
                    binary = np.zeros(label.shape, dtype='uint8')
                    binary[ind] = 1
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                    bbox = contours[0] * scale

                bbox = bbox.astype('int32')
                bboxes.append(bbox.reshape(-1))
                scores.append(score_i)

            outputs.update(dict(bboxes=bboxes, scores=scores))
            
            write_result_ctw(img_name, outputs, config['test_cfg']['result_text_path'])
            # print(f'writeing res for {img_name}')
            ori_img = cv2.imread(img_path)
            boxes = [b.reshape(-1, 2) for b in outputs['bboxes']]
            vis_img = cv2.polylines(ori_img, boxes, True, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(config['test_cfg']['result_img_path'], img_name), vis_img)
            # print(f'writeing res img for {img_name}')

