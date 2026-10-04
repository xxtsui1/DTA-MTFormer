import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils_wt

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path
from utils_img import save_image


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    if np.sum(y_pred) > 0.5:
        y_pred_l=1
    else:
        y_pred_l=0    
    if y_pred_l==y_true:
        ap_i=1
    else:
        ap_i=0    
    AP.append(ap_i)
    return AP



def compute_dice(pred, gt):   
    AP = [] 
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    pred_my = np.zeros(pred.shape)
    if gt.sum()==0:
        gt[gt==0]=1
        pred_my [pred<0.5]=1
    else:
        pred_my [pred>0.5]=1
    dice_this = 2 * np.sum(gt * pred_my) / (np.sum(gt) + np.sum(pred_my)+0.0000001)  
    AP.append(dice_this)          
    return AP 


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils_wt.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils_wt.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets, mask_gt in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)
        mask_gt = mask_gt.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs, patch_outputs, mask_loss,masks,c = model(samples)
            loss_fn=nn.BCEWithLogitsLoss()
            loss = loss_fn(outputs, targets.float())
            #loss = F.multilabel_soft_margin_loss(outputs, targets)
            metric_logger.update(cls_loss=loss.item())            

            ploss = loss_fn(patch_outputs, targets.float())
            metric_logger.update(pat_loss=ploss.item())
            loss = loss + ploss

            aloss = mask_loss
            metric_logger.update(aloss=mask_loss.item())
            loss = loss + aloss

            seg_fn = nn.BCEWithLogitsLoss(reduction='none')
            # 模型输出 masks 尺寸可能大于 GT mask，将 GT 上采样到相同尺寸
            if mask_gt.shape[-2:] != masks.shape[-2:]:
                mask_gt = F.interpolate(mask_gt, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss_seg = torch.mean(seg_fn(masks, mask_gt))
            metric_logger.update(seg_loss=loss_seg.item())
            loss = loss + loss_seg

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    mAP = []
    patch_mAP = []
    attn_mAP=[]
    seg_mAP = []

    metric_logger = utils_wt.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target,mask_gt in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask_gt = mask_gt.to(device, non_blocking=True)
        batch_size = images.shape[0]

        output, patch_output, attn_output,mask,_ = model(images)

        output = torch.sigmoid(output)
        mAP_list = compute_mAP(target, output)
        mAP = mAP + mAP_list
        metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)

        patch_output = torch.sigmoid(patch_output)        
        mAP_list = compute_mAP(target, patch_output)
        patch_mAP = patch_mAP + mAP_list
        metric_logger.meters['patch_mAP'].update(np.mean(mAP_list), n=batch_size)

        attn_output = torch.sigmoid(attn_output)
        mAP_list = compute_mAP(target, attn_output)
        attn_mAP = attn_mAP + mAP_list
        metric_logger.meters['attn_mAP'].update(np.mean(mAP_list), n=batch_size)

        mask = torch.sigmoid(mask)
        # 将预测 mask 下采样到 GT 尺寸以计算 dice
        if mask.shape[-2:] != mask_gt.shape[-2:]:
            mask = F.interpolate(mask, size=mask_gt.shape[-2:], mode='bilinear', align_corners=False)
        seg_list =  compute_dice(mask,mask_gt)
        seg_mAP = seg_mAP+seg_list
        metric_logger.meters['seg_mAP'].update(np.mean(seg_list), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        '* mAP {mAP.global_avg:.3f} patch_mAP {patch_mAP.global_avg:.3f} attn_mAP {attn_mAP.global_avg:.3f} seg_mAP {seg_mAP.global_avg:.3f} '
        .format(mAP=metric_logger.mAP, patch_mAP=metric_logger.mAP,attn_mAP=metric_logger.attn_mAP,seg_mAP=metric_logger.seg_mAP))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args, epoch=None):
    metric_logger = utils_wt.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()

    img_list = open(os.path.join('busi_valid.txt')).readlines()
    index = 0
    for image_list, target in metric_logger.log_every(data_loader, 1, header):
        images1 = image_list.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        img_name = img_list[index].strip()
        index += args.world_size

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = img_temp[:, :, :, 0] 
        orig_images[:, :, :, 1] = img_temp[:, :, :, 1] 
        orig_images[:, :, :, 2] = img_temp[:, :, :, 2] 

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]

        with torch.cuda.amp.autocast():
            cam_list = []
            for s in range(len(image_list)):
                images = image_list.to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[
                    3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                output, cams, patch_attn = model(images, return_att=True, attention_type=args.attention_type)
                patch_attn = torch.sum(patch_attn, dim=0)

                if args.patch_attn_refine:
                    cams = torch.matmul(patch_attn.unsqueeze(1),
                                                  cams.view(cams.shape[0], cams.shape[1],
                                                                      -1, 1)).reshape(cams.shape[0],
                                                                                      cams.shape[1],
                                                                                      w_featmap, h_featmap)

                cams = \
                    F.interpolate(cams, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cams = cams * target.clone().view(args.nb_classes, 1, 1)

                cam_list.append(cams)

            sum_cam = torch.sum(torch.stack(cam_list), dim=0)
            sum_cam = sum_cam.unsqueeze(0)
            
            if torch.sigmoid(output).item()>0.5:
                    pred = 1
            else:
                    pred=0
            precision =int (pred==target.item())
            
            for b in range(images.shape[0]):
                cls_attention = sum_cam[b,:,:,:]

                cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                cls_attention = cls_attention
                #save_image(cls_attention, str(precision)+"_"+img_name.split('')[0])
                save_image(cls_attention, str(precision)+"_"+img_name.split('.png')[0]+".png")

        

def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)
