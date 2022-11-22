#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
# import pdb
from os.path import exists, join, split
import threading
from datetime import datetime

import time

import numpy as np
import shutil

from tqdm import tqdm
import sys
from PIL import Image
import torch
from torch import nn, normal
import torch.backends.cudnn as cudnn
from torch.nn.modules import transformer
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from min_norm_solvers import MinNormSolver
import torch.nn.functional as F

import data_transforms as d_transforms
from model.models import DPTSegmentationModel, DPTSegmentationModelMultiHead, CerberusSegmentationModelMultiHead, DPTNormalModel
from model.transforms import PrepareForNet
from common import NYU40_PALETTE, CITYSCAPE_PALETTE, AFFORDANCE_PALETTE
from utils import *
from dataset import *
from losses import compute_loss


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, filename='./'+ datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TASK = 'SEGMENTATION'  # 'ATTRIBUTE', 'AFFORDANCE', 'SEGMENTATION' 
TRANSFER_FROM_TASK = None  #'ATTRIBUTE', 'AFFORDANCE', 'SEGMENTATION', or None to unable transfer

task_list = None
middle_task_list = None

if TASK == 'ATTRIBUTE':
    task_list = ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']
    FILE_DESCRIPTION = '_attribute'
    PALETTE = AFFORDANCE_PALETTE
    EVAL_METHOD = 'mIoU'
elif TASK == 'AFFORDANCE':
    task_list = ['L','M','R','S','W']
    FILE_DESCRIPTION = '_affordance'
    PALETTE = AFFORDANCE_PALETTE
    EVAL_METHOD = 'mIoU'
elif TASK =='SEGMENTATION':
    task_list = ['Segmentation']
    FILE_DESCRIPTION = ''
    PALETTE = NYU40_PALETTE
    EVAL_METHOD = 'mIoUAll'
elif TASK =='NORMAL':
    task_list = ['NORMAL']
    FILE_DESCRIPTION = ''
    EVAL_METHOD = 'RMSE'
else:
    task_list = None
    FILE_DESCRIPTION = ''
    PALETTE = None
    EVAL_METHOD = None

if TRANSFER_FROM_TASK == 'ATTRIBUTE':
    middle_task_list = ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']
elif TRANSFER_FROM_TASK == 'AFFORDANCE':
    middle_task_list = ['L','M','R','S','W']
elif TRANSFER_FROM_TASK =='SEGMENTATION':
    middle_task_list = ['Segmentation']
elif TRANSFER_FROM_TASK is None:
    pass


# if TRANSFER_FROM_TASK is not None:
#     TENSORBOARD_WRITER = SummaryWriter(comment='From_'+TRANSFER_FROM_TASK+'_TO_'+TASK)
# elif TASK is not None:
#     TENSORBOARD_WRITER = SummaryWriter(comment=TASK)
# else:
#     TENSORBOARD_WRITER = SummaryWriter(comment='Nontype')
TENSORBOARD_WRITER = SummaryWriter("./exp_logs")
def downsampling(x, size=None, scale=None, mode='nearest'):
    if size is None:
        size = (int(scale * x.size(2)) , int(scale * x.size(3)))
    h = torch.arange(0,size[0]) / (size[0] - 1) * 2 - 1
    w = torch.arange(0,size[1]) / (size[1] - 1) * 2 - 1
    grid = torch.zeros(size[0] , size[1] , 2)
    grid[: , : , 0] = w.unsqueeze(0).repeat(size[0] , 1)
    grid[: , : , 1] = h.unsqueeze(0).repeat(size[1] , 1).transpose(0 , 1)
    grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    if x.is_cuda:
        grid = grid.cuda()
    return torch.nn.functional.grid_sample(x , grid , mode = mode)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def validate_normal(val_loader, model, epoch):
    print("=============validation begin==============")
    val_loader = tqdm(val_loader, desc="Loop: Validation")
    model.eval()
    with torch.no_grad():
        total_normal_errors = None
        
        for data_dict in val_loader:

            # data to device
            img = data_dict['img'].cuda()
            gt_norm = data_dict['norm'].cuda()
            gt_norm_mask = data_dict['norm_valid_mask'].cuda()

            # forward pass

            norm_out_list, _, _ = model(img, gt_norm_mask=gt_norm_mask, mode='val')
            norm_out = norm_out_list[-1]

            # upsample if necessary
            if norm_out.size(2) != gt_norm.size(2):
                norm_out = F.interpolate(norm_out, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)

            pred_norm = norm_out[:, :3, :, :]  # (B, 3, H, W)
            pred_kappa = norm_out[:, 3:, :, :]  # (B, 1, H, W)

            prediction_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
            prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
            E = torch.acos(prediction_error) * 180.0 / np.pi

            mask = gt_norm_mask[:, 0, :, :]
            if total_normal_errors is None:
                total_normal_errors = E[mask]
            else:
                total_normal_errors = torch.cat((total_normal_errors, E[mask]), dim=0)

        total_normal_errors = total_normal_errors.data.cpu().numpy()
        metrics = compute_normal_errors(total_normal_errors)
        log_normal_errors(metrics, 'log/test.txt', first_line='epoch: {}'.format(epoch + 1))

        

        return metrics

def validate(val_loader, model, criterion, eval_score=None, print_freq=10, epoch=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_array = list()
    for it in task_list:
        losses_array.append(AverageMeter())
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            
            target_var = list()
            for idx in range(len(target)):
                target[idx] = target[idx].cuda(non_blocking=True)
                target_var.append(torch.autograd.Variable(target[idx], volatile=True))
            

            # compute output
            
            output, _ = model(input_var)
            softmaxf = nn.LogSoftmax()

            loss_array = list()
            for idx in range(len(output)):
                output[idx] = softmaxf(output[idx])
                loss_array.append(criterion(output[idx],target_var[idx]))

            loss = sum(loss_array)

            # measure accuracy and record loss

            losses.update(loss.item(), input.size(0))

            for idx, it in enumerate(task_list):
                (losses_array[idx]).update((loss_array[idx]).item(), input.size(0))

            scores_array = list()

            for idx in range(len(output)):
                scores_array.append(eval_score(output[idx], target_var[idx]))
            
            score.update(np.nanmean(scores_array), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))
            
    TENSORBOARD_WRITER.add_scalar('val_loss_average', losses.avg, global_step=epoch)
    TENSORBOARD_WRITER.add_scalar('val_score_average', score.avg, global_step=epoch)

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg

def validate_cerberus(val_loader, model, criterion, eval_score=None, print_freq=10, epoch=None):
    
    task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],
                       ['L','M','R','S','W'],
                       ['Segmentation']] 
    
    batch_time_list = list()
    losses_list = list()
    losses_array_list = list()
    score_list = list()
    score = AverageMeter()

    for i in range(3):
        batch_time_list.append(AverageMeter())
        losses_list.append(AverageMeter())
        losses_array = list()
        for it in task_list_array[i]:
            losses_array.append(AverageMeter())
        losses_array_list.append(losses_array)
        score_list.append(AverageMeter())

    # switch to evaluate mode
    model.eval()
    # if transfer_model is not None:
    #     transfer_model.eval()

    end = time.time()
    for i, pairs in enumerate(val_loader):
        for index, (input,target) in enumerate(pairs):
            with torch.no_grad():
                input = input.cuda()
                input_var = torch.autograd.Variable(input, volatile=True)
                
                target_var = list()
                for idx in range(len(target)):
                    target[idx] = target[idx].cuda(non_blocking=True)
                    target_var.append(torch.autograd.Variable(target[idx], volatile=True))
                

                # compute output
                output, _, _ = model(input_var, index)
                softmaxf = nn.LogSoftmax()

                loss_array = list()
                for idx in range(len(output)):
                    output[idx]= softmaxf(output[idx])
                    loss_array.append(criterion(output[idx],target_var[idx]))

                loss = sum(loss_array)

                # measure accuracy and record loss

                losses_list[index].update(loss.item(), input.size(0))

                for idx, it in enumerate(task_list_array[index]):
                    (losses_array_list[index][idx]).update((loss_array[idx]).item(), input.size(0))

                scores_array = list()

                if index < 2:
                    for idx in range(len(output)):
                        scores_array.append(eval_score(output[idx], target_var[idx]))
                elif index == 2:
                    for idx in range(len(output)):
                        scores_array.append(mIoUAll(output[idx], target_var[idx]))
                else:
                    assert 0 == 1
                
                tmp = np.nanmean(scores_array)
                if not np.isnan(tmp):
                    score_list[index].update(tmp, input.size(0))
                else:
                    pass

            # measure elapsed time
            batch_time_list[index].update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Score {score.val:.3f} ({score.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time_list[index], loss=losses_list[index],
                    score=score_list[index]))
        score.update(np.nanmean([score_list[0].val, score_list[1].val, score_list[2].val]))
        if i % print_freq == 0:
            logger.info('total score is:{score.val:.3f} ({score.avg:.3f})'.format(
                score = score
            ))
    
    for idx, item in enumerate(['attribute','affordance','segmentation']):
        TENSORBOARD_WRITER.add_scalar('val_'+ item +'_loss_average', losses_list[idx].avg, global_step=epoch)
        TENSORBOARD_WRITER.add_scalar('val_'+ item +'_score_average', score_list[idx].avg, global_step=epoch)

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))
    TENSORBOARD_WRITER.add_scalar('val_score_average', score.avg, global_step=epoch)


    return score.avg

def train_normal(args, train_loader, model, criterion, optimizer, epoch):

    train_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train Normal")
    model.train()
    for i, data_dict in enumerate(train_loader):
        
        img = data_dict['img'].cuda()
        gt_norm = data_dict['norm'].cuda()
        gt_norm_mask = data_dict['norm_valid_mask'].cuda()

        _, pred_list, coord_list = model(img, gt_norm_mask=gt_norm_mask, mode='train')
        # print(pred_list, coord_list, gt_norm, gt_norm_mask)
        loss = criterion(pred_list, coord_list, gt_norm, gt_norm_mask)
        # print(loss, len(loss))
        loss_ = float(loss.data.cpu().numpy())
        train_loader.set_description(f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train. Loss: {'%.5f' % loss_}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_iter = epoch * len(train_loader) + i
        TENSORBOARD_WRITER.add_scalar('train_normal_loss', loss.item(), total_iter)

def train_seg(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_array = list()
    for it in task_list:
        losses_array.append(AverageMeter())
    scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        input_var = torch.autograd.Variable(input)

        target_var = list()
        for idx in range(len(target)):
            target[idx] = target[idx].cuda()
            target_var.append(torch.autograd.Variable(target[idx]))

        # compute output

        output, _ = model(input_var)


        softmaxf = nn.LogSoftmax()
        loss_array = list()

        assert len(output) == len(target)

        for idx in range(len(output)):
            output[idx] = softmaxf(output[idx])
            loss_array.append(criterion(output[idx],target_var[idx]))

        loss = sum(loss_array)

        # measure accuracy and record loss

        losses.update(loss.item(), input.size(0))

        for idx, it in enumerate(task_list):
            (losses_array[idx]).update((loss_array[idx]).item(), input.size(0))

        scores_array = list()

        for idx in range(len(output)):
            scores_array.append(eval_score(output[idx], target_var[idx]))
        
        scores.update(np.nanmean(scores_array), input.size(0))

        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            losses_info = ''
            for idx, it in enumerate(task_list):
                losses_info += 'Loss_{0} {loss.val:.4f} ({loss.avg:.4f})\t'.format(it, loss=losses_array[idx])
                TENSORBOARD_WRITER.add_scalar('train_task_' + it + '_loss_val', losses_array[idx].val, 
                    global_step= epoch * len(train_loader) + i)
                TENSORBOARD_WRITER.add_scalar('train_task_' + it + '_loss_average', losses_array[idx].avg,
                    global_step= epoch * len(train_loader) + i)

            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        '{loss_info}'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,loss_info=losses_info,
                top1=scores))
            
            TENSORBOARD_WRITER.add_scalar('train_loss_val', losses.val, global_step= epoch * len(train_loader) + i)
            TENSORBOARD_WRITER.add_scalar('train_loss_average', losses.avg, global_step= epoch * len(train_loader) + i)
            TENSORBOARD_WRITER.add_scalar('train_scores_val', scores.val, global_step= epoch * len(train_loader) + i)
            TENSORBOARD_WRITER.add_scalar('train_scores_val', scores.avg, global_step= epoch * len(train_loader) + i)

    TENSORBOARD_WRITER.add_scalar('train_epoch_loss_average', losses.avg, global_step= epoch)
    TENSORBOARD_WRITER.add_scalar('train_epochscores_val', scores.avg, global_step= epoch)

def train_cerberus(train_loader, model, criterion, optimizer, epoch, 
          eval_score=None, print_freq=1):
    
    task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],
                       ['L','M','R','S','W'],
                       ['Segmentation']]

    root_task_list_array = ['At', 'Af', 'Seg']

    batch_time_list = list()
    data_time_list = list()
    losses_list = list()
    losses_array_list = list()
    scores_list = list()
    
    for i in range(3):
        batch_time_list.append(AverageMeter())
        data_time_list.append(AverageMeter())
        losses_list.append(AverageMeter())
        losses_array = list()
        for it in task_list_array[i]:
            losses_array.append(AverageMeter())
        losses_array_list.append(losses_array)
        scores_list.append(AverageMeter())

    model.train()

    end = time.time()

    moo = True

    for i, in_tar_name_pair in enumerate(train_loader):
        if moo :
            grads = {}
        task_loss_array = []
        for index, (input, target, name) in enumerate(in_tar_name_pair):
            # measure data loading time
            data_time_list[index].update(time.time() - end)

            if moo:

                input = input.cuda()
                input_var = torch.autograd.Variable(input)

                target_var = list()
                for idx in range(len(target)):
                    target[idx] = target[idx].cuda()
                    target_var.append(torch.autograd.Variable(target[idx]))

                # compute output
                output, _, _ = model(input_var, index)
                
                # if transfer_model is not None:
                #     output = transfer_model(output)
                softmaxf = nn.LogSoftmax()
                loss_array = list()

                assert len(output) == len(target)

                for idx in range(len(output)):
                    output[idx] = softmaxf(output[idx])
                    loss_raw = criterion(output[idx],target_var[idx])
                    
                    loss_enhance = loss_raw 

                    if torch.isnan(loss_enhance):
                        print("nan")
                        logger.info('loss_raw is: {0}'.format(loss_raw))
                        logger.info('loss_enhance is: {0}'.format(loss_enhance))
                        exit(0)
                        # loss_array.append(loss_enhance)
                    else:
                        loss_array.append(loss_enhance)

                    local_loss = sum(loss_array)
                    local_loss_enhance = local_loss 

                # backward for gradient calculate
                for cnt in model.pretrained.parameters():
                    cnt.grad = None
                model.scratch.layer1_rn.weight.grad = None
                model.scratch.layer2_rn.weight.grad = None
                model.scratch.layer3_rn.weight.grad = None
                model.scratch.layer4_rn.weight.grad = None

                local_loss_enhance.backward()

                grads[root_task_list_array[index]] = []
                for par_name, cnt in model.pretrained.named_parameters():
                    if cnt.grad is not None:
                        grads[root_task_list_array[index]].append(Variable(cnt.grad.data.clone(),requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer1_rn.weight.grad.data.clone(), requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer2_rn.weight.grad.data.clone(), requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer3_rn.weight.grad.data.clone(), requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer4_rn.weight.grad.data.clone(), requires_grad = False))
            else:
                pass
            if moo: 
                if torch.isnan(local_loss_enhance):
                    print("nan")
                    logger.info('loss_raw is: {0}'.format(local_loss))
                    logger.info('loss_enhance is: {0}'.format(local_loss_enhance))
                    exit(0)
                    # loss_array.append(loss_enhance)
                else:
                    task_loss_array.append(local_loss_enhance)

                # measure accuracy and record loss

                losses_list[index].update(local_loss_enhance.item(), input.size(0))

                for idx, it in enumerate(task_list_array[index]):
                    (losses_array_list[index][idx]).update((loss_array[idx]).item(), input.size(0))

                scores_array = list()

                if index < 2:
                    for idx in range(len(output)):
                        scores_array.append(eval_score(output[idx], target_var[idx]))
                elif index == 2:
                    for idx in range(len(output)):
                        scores_array.append(mIoUAll(output[idx], target_var[idx]))
                else:
                    assert 0 == 1
                

                scores_list[index].update(np.nanmean(scores_array), input.size(0))

            # compute gradient and do SGD step
            if index == 2:
                if moo:
                    del input, target, input_var, target_var
                    task_loss_array_new = []
                    for index_new, (input_new, target_new, _) in enumerate(in_tar_name_pair):
                        input_var_new = torch.autograd.Variable(input_new.cuda())
                        target_var_new = [torch.autograd.Variable(target_new[idx].cuda()) for idx in range(len(target_new))]
                        output_new, _, _ = model(input_var_new, index_new)
                        loss_array_new = [criterion(softmaxf(output_new[idx]),target_var_new[idx]) \
                            for idx in range(len(output_new))]
                        local_loss_new = sum(loss_array_new)
                        task_loss_array_new.append(local_loss_new)
                    assert len(task_loss_array_new) == 3
                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[cnt] for cnt in root_task_list_array])

                    logger.info('scale is: |{0}|\t|{1}|\t|{2}|\t'.format(sol[0], sol[1], sol[2]))
                    
                    loss_new = 0
                    loss_new = sol[0] * task_loss_array_new[0] + sol[1] * task_loss_array_new[1] \
                         + sol[2] * task_loss_array_new[2]
                    
                    optimizer.zero_grad()
                    loss_new.backward()
                    optimizer.step()
                else:
                    assert len(task_loss_array) == 3

                    loss = sum(task_loss_array)
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
            
            if moo:
            # measure elapsed time
                batch_time_list[index].update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    losses_info = ''
                    for idx, it in enumerate(task_list_array[index]):
                        losses_info += 'Loss_{0} {loss.val:.4f} ({loss.avg:.4f}) \t'.format(it, loss=losses_array_list[index][idx])
                        TENSORBOARD_WRITER.add_scalar('train_task_'+ it +'_loss_val', losses_array_list[index][idx].val,
                            global_step= epoch * len(train_loader) + i)
                        TENSORBOARD_WRITER.add_scalar('train_task_'+ it +'_loss_avg', losses_array_list[index][idx].avg,
                            global_step= epoch * len(train_loader) + i)

                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                '{loss_info}'
                                'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time_list[index],
                        data_time=data_time_list[index], loss=losses_list[index],loss_info=losses_info,
                        top1=scores_list[index]))
                    logger.info('File name is: {}'.format(','.join(name)))
                    TENSORBOARD_WRITER.add_scalar('train_'+ str(index) +'_losses_val', losses_list[index].val,
                        global_step= epoch * len(train_loader) + i)
                    TENSORBOARD_WRITER.add_scalar('train_'+ str(index) +'_losses_avg', losses_list[index].avg,
                        global_step= epoch * len(train_loader) + i)
                    TENSORBOARD_WRITER.add_scalar('train_'+ str(index) +'_score_val', scores_list[index].val,
                        global_step= epoch * len(train_loader) + i)
                    TENSORBOARD_WRITER.add_scalar('train_'+ str(index) +'_score_avg', scores_list[index].avg,
                        global_step= epoch * len(train_loader) + i)
    for i in range(3):
        TENSORBOARD_WRITER.add_scalar('train_epoch_loss_average', losses_list[index].avg, global_step= epoch)
        TENSORBOARD_WRITER.add_scalar('train_epoch_scores_val', scores_list[index].avg, global_step= epoch)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def train_single(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)
    
    if args.task == "seg":
        single_model = DPTSegmentationModel(args.classes, backbone="vitb_rn50_384")

    elif args.task == "normal":
        single_model = DPTNormalModel(args.classes, backbone="vitb_rn50_384", sampling_ratio=0.4, importance_ratio=0.7)
    
    model = single_model.cuda()

    if args.task == "seg":
        criterion = nn.NLLLoss2d(ignore_index=255)
    elif args.task == "normal":
        criterion = compute_loss(args)
    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = d_transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []

    if args.random_rotate > 0:
        t.append(d_transforms.RandomRotateMultiHead(args.random_rotate))
    if args.random_scale > 0:
        t.append(d_transforms.RandomScaleMultiHead(args.random_scale))
    t.extend([d_transforms.RandomCropMultiHead(crop_size),
                d_transforms.RandomHorizontalFlipMultiHead(),
                d_transforms.ToTensorMultiHead(),
                normalize])
    if args.task == "normal": 
        train_loader = torch.utils.data.DataLoader(
            Normal_Single(args, data_dir, 'train'),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            Normal_Single(args, data_dir, 'val'),
            batch_size=1, shuffle=False, num_workers=num_workers,
            pin_memory=True
        )

    elif args.task == "seg":       
        train_loader = torch.utils.data.DataLoader(
            Seg_Single(data_dir, 'train', d_transforms.Compose(t)),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
                Seg_Single(data_dir, 'val', d_transforms.Compose([
                d_transforms.RandomCropMultiHead(crop_size),
                d_transforms.ToTensorMultiHead(),
                normalize,
            ])),
            batch_size=1, shuffle=False, num_workers=num_workers,
            pin_memory=True, drop_last=True
        )


    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        if args.task == "seg":
            train_seg(train_loader, model, criterion, optimizer, epoch, eval_score=eval(EVAL_METHOD))
            prec1 = validate(val_loader, model, criterion, eval_score=eval(EVAL_METHOD), epoch=epoch)
        elif args.task =="normal": 
            train_normal(args, train_loader, model, criterion, optimizer, epoch) 
            metrics = validate_normal(val_loader, model, epoch)
            prec1 = metrics["a1"]
            
        # evaluate on validation set
            
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = './exp_logs/models/model_best.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)

def train_seg_cerberus(args):

    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)
    
    single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
    model = single_model.cuda()

    criterion = nn.NLLLoss2d(ignore_index=255)
    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))


    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []

    if args.random_rotate > 0:
        t.append(transforms.RandomRotateMultiHead(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScaleMultiHead(args.random_scale))
    t.extend([transforms.RandomCropMultiHead(crop_size),
                transforms.RandomHorizontalFlipMultiHead(),
                transforms.ToTensorMultiHead(),
                normalize])

    dataset_at_train = Seg_Single(data_dir, 'train_attribute', transforms.Compose(t), out_name=True)
    dataset_af_train = Seg_Single(data_dir, 'train_affordance', transforms.Compose(t), out_name=True)
    dataset_seg_train = Seg_Single(data_dir, 'train', transforms.Compose(t), out_name=True)
    # cerberus_dataset = ConcatSegList(dataset_at_train, dataset_af_train, dataset_seg_train)

    # cerberus_sampler = torch.utils.data.distributed.DistributedSampler(cerberus_dataset)
    # train_at_sampler = torch.utils.data.distributed.DistributedSampler(dataset_at_train)
    # train_acerberus_samplerf_sampler = torch.utils.data.distributed.DistributedSampler(dataset_af_train)
    # train_seg_sampler = torch.utils.data.distributed.DistributedSampler(dataset_seg_train)

    # train_loader = (torch.utils.data.DataLoader(
    #     ConcatSegList(dataset_at_train, dataset_af_train, dataset_seg_train),
    #      batch_size=batch_size, num_workers=num_workers,
    #         pin_memory=True, drop_last=True
    # ))
        # train_set,batch_size=batch_size, num_workers=num_workers,
        # pin_memory=True, drop_last=True, sampler=train_sampler
    train_loader = (torch.utils.data.DataLoader(
        ConcatSegList(dataset_at_train, dataset_af_train, dataset_seg_train),
         batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True
    ))

    dataset_at_val = Seg_Single(data_dir, 'val_attribute', transforms.Compose([
                transforms.RandomCropMultiHead(crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))
    dataset_af_val = Seg_Single(data_dir, 'val_affordance', transforms.Compose([
                transforms.RandomCropMultiHead(crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))
    dataset_seg_val = Seg_Single(data_dir, 'val', transforms.Compose([
                transforms.RandomCropMultiHead(crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))

    val_loader = (torch.utils.data.DataLoader(
        ConcatSegList(dataset_at_val, dataset_af_val, dataset_seg_val),
        batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True
    ))

    # define loss function (criterion) and optimizer,
    optimizer = torch.optim.SGD([
                                # {'params':model.parameters()}],
                                {'params':model.pretrained.parameters()},
                                {'params':model.scratch.parameters()}],
                                # {'params':single_model.sigma.parameters(), 'lr': args.lr * 0.01}],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                if name[:5] == 'sigma':
                    model.state_dict()[name].copy_(param)
                else:
                    # model.state_dict()[name].copy_(param)
                    pass
            print("=> loaded sigma checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            for name, param in checkpoint['state_dict'].items():
                if name[:5] == 'sigma':
                    pass
                    # model.state_dict()[name].copy_(param)
                else:
                    model.state_dict()[name].copy_(param)
                    # pass
            print("=> loaded model checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate_cerberus(val_loader, model, criterion, eval_score=mIoU, epoch=0)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

        train_cerberus(train_loader, model, criterion, optimizer, epoch, eval_score=mIoU)
        
        #if epoch%10==1:
        prec1 = validate_cerberus(val_loader, model, criterion, eval_score=mIoU, epoch=epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = 'checkpoint_latest.pth.tar'        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % 10 == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['single', 'multi'])
    parser.add_argument('-d', '--data-dir', default='../dataset/nyud2')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--pretrained-model', dest='pretrained_model',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)

    parser.add_argument("--task", type=str, default="normal")
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)
    parser.add_argument("--data_augmentation_color", default=True, action="store_true")
    parser.add_argument("--data_augmentation_hflip", default=True, action="store_true")
    parser.add_argument("--data_augmentation_random_crop", default=False, action="store_true")
    parser.add_argument('--loss_fn', default='UG_NLL_ours', type=str, help='{L1, L2, AL, NLL_vMF, NLL_ours, UG_NLL_vMF, UG_NLL_ours}')


    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    return args

def main():
    args = parse_args()
    if args.cmd == 'single':
        train_single(args)
    elif args.cmd == "multi":
        train_cerberus(args)

if __name__ == '__main__':
    main()