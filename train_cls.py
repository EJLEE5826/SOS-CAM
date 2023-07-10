import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append(os.getcwd())

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import argparse
from PIL import Image 
import datetime as dt

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from models.resnet50_SOS import resnet50_sos
from models.resnet50 import resnet50_base

from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
from utils.LoadData import train_data_loader, valid_data_loader
from utils.Metrics import Cls_Accuracy, get_map_metrics
from utils.util import output_visualize
from tqdm import tqdm
import wandb 


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_arguments():
    parser = argparse.ArgumentParser(description='SOS-CAM pytorch implementation')

    parser.add_argument("--wandb_name", type=str, default='', help='wandb name')

    parser.add_argument('-v', '--val', action='store_true',
                        help='evaluation mode')

    parser.add_argument("--img_dir", type=str, help='Directory of training images')
    parser.add_argument("--train_list", type=str, help='Path to training list')
    parser.add_argument("--test_list", type=str, help='Path to test list')
    parser.add_argument('--save_folder', default='checkpoints/resnet_sos_baseline', help='Location to save checkpoint models')
    parser.add_argument("--checkpoint", type=str, default=None, help='Checkpoint path. somewhere/best.pth')

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=768)
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--shuffle_val", action='store_false')

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.25, help='object cues for the pseudo seg map generation')

    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_interval', default=5, type=int, help='interval of save checkpoint models')
    parser.add_argument("--global_counter", type=int, default=0)

    return parser.parse_args()

def get_model(args):

    if 'resnet' in args.save_folder:
        if 'sos' in args.save_folder:
            model = resnet50_sos()
        else:
            model = resnet50_base()
    else:
        raise Exception('No model is selected') 
    
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)


    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.get_parameter_groups()
    
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], 
        momentum=0.9, 
        weight_decay=args.weight_decay, 
        nesterov=True
    )

    return  model, optimizer


def validate():
    print('\nvalidating ... ', flush=True, end='')
    cls_acc_matrix = Cls_Accuracy()
    val_loss = AverageMeter()
    
    model.eval()

    gt_list = []
    preds_probs = []
    
    with torch.no_grad():
        for idx, dat in enumerate(tqdm(val_loader)):
            img, label, _ = dat
            
            B, _, H, W = img.size()
            
            label = label.to('cuda', non_blocking=True)
            img = img.to('cuda', non_blocking=True)
            
            logit, cam = model(img, label)

            """ classification loss """
            loss = F.multilabel_soft_margin_loss(logit, label)
            cls_acc_matrix.update(logit, label)

            val_loss.update(loss.data.item(), img.size()[0])

            gt_list.append(label.cpu().numpy())
            preds_probs.append(logit.cpu().numpy())
            
            """ obtain CAMs """
            cam = cam.cpu().detach()
            cam_norm, _ = torch.max(cam, 1, keepdim=True)

            cam = cam.numpy()
                            
            """ segmentation label generation """
            cam[cam < args.alpha] = 0  # object cue
            bg = np.zeros((B, 1, H, W), dtype=np.float32)

            agg_cam = torch.tensor(cam.copy(), requires_grad=False)
            cam = cam_norm.numpy()
            cam[cam < args.alpha] = 0 

            agg_cam, _ = torch.max(agg_cam, 1, keepdim=True)
            agg_cam = agg_cam.numpy()
            agg_cam[agg_cam < args.alpha] = 0

            agg_pred_map = np.concatenate([bg, agg_cam], axis=1) 
            pred_map = np.concatenate([bg, cam], axis=1)  # [B, 7, H, W]
            pred_map = pred_map.argmax(1) # channel-level maximum 
            agg_pred_map = agg_pred_map.argmax(1)

    
    """ validation score """
    val_cls_acc, val_precision, val_recall, val_f1 = cls_acc_matrix.compute_avg_acc()

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    valid_mAP, valid_ap = get_map_metrics(gt_label, preds_probs)
    
    
    """ visualization with wandb """
    results = []
    result_vis, label_list = output_visualize(img[0], cam[0], agg_cam[0], label[0], pred_map[0], agg_pred_map[0])

    for i in range(result_vis.shape[0]):
        vis = np.transpose(result_vis[i], (1, 2, 0)) * 255
        vis = vis.astype(np.uint8)
        image = Image.fromarray(vis).convert('RGB')
        results.append(image)

    wandb.log({ 
            'Result Visualization' : [wandb.Image(image, caption=str(i)) for i, image in enumerate(results)], 
            })
    
    """ print results """
    print('validating loss: %.4f' % val_loss.avg)
    print('validating acc: %.4f' % val_cls_acc)
    print('validating f1(per cls, samples):', val_f1)
    print('validating precision:', val_recall)
    print('validating recall:', val_precision)
    print(f'validating mAP, AP: {valid_mAP:.4f}, {valid_ap}')
    
    
    return val_loss.avg, val_cls_acc, val_precision, val_recall, val_f1, valid_mAP, valid_ap
    

def train(current_epoch):
    train_loss = AverageMeter()
    cls_acc_matrix = Cls_Accuracy()

    model.train()
    
    global_counter = args.global_counter

    """ learning rate decay """
    res = reduce_lr(args, optimizer, current_epoch)

    for idx, dat in enumerate(train_loader):

        img, label, _ = dat
        label = label.to('cuda', non_blocking=True)
        img = img.to('cuda', non_blocking=True)

        logit = model(img)
        """ classification loss """
        loss = F.multilabel_soft_margin_loss(logit, label)

        """ backprop """
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        cls_acc_matrix.update(logit, label)
        
        train_loss.update(loss.data.item(), img.size()[0])

        global_counter += 1

        """ tensorboard log """
        if global_counter % args.show_interval == 0:
            train_cls_acc = cls_acc_matrix.compute_avg_acc()

            print('Epoch: [{}][{}/{}]\t'
                  'LR: {:.5f}\t'
                  'ACC: {:.5f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    current_epoch, idx+1, len(train_loader),
                    optimizer.param_groups[0]['lr'], 
                    train_cls_acc[0], loss=train_loss,
                    )
                 )

    args.global_counter = global_counter

    return train_cls_acc[0], train_loss.val, train_loss.avg

if __name__ == '__main__':
    args = get_arguments()
    
    nGPU = torch.cuda.device_count()
    print("start training the classifier, nGPU = %d" % nGPU)
    
    args.batch_size *= nGPU
    args.num_workers *= nGPU

    if args.val:
        args.epoch = 1
    
    print('Running parameters:\n', args)
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    train_loader = train_data_loader(args)
    val_loader = valid_data_loader(args)
    print('# of train dataset:', len(train_loader) * args.batch_size)
    print('# of valid dataset:', len(val_loader) * args.batch_size)
    print()

    best_score = 0
    model, optimizer = get_model(args)
    criterion = nn.MultiLabelSoftMarginLoss()

    # wandb setting
    wandb.init()
    wandb.run.name = args.wandb_name 
    wandb.config.update(args)
    wandb.watch(model)

    x = dt.datetime.now()

    total_results_index = ['Train Acc', 'Train Avg Loss', 'Val ACC', 'Val Avg Loss', 
                           'Val Precision_per class', 'Val Precision_per sample', 
                           'Val Recall_per class', 'Val Recall_per sample', 
                           'Val F1 Score_per class ', 'Val F1 Score_per sample', 
                           'Val mAP', 'Val AP'
                          ]
    
    for current_epoch in range(1, args.epoch+1):
        
        if not args.val:
            train_cls_acc, loss, train_avg_loss = train(current_epoch)
        results = validate()
        val_avg_loss, val_cls_acc, val_precision, val_recall, val_f1, valid_mAP, valid_ap = results
        
        total_results_value = [train_cls_acc, train_avg_loss, val_cls_acc, val_avg_loss,
                               val_precision[0].mean(), val_precision[1],
                               val_recall[0].mean(), val_recall[1],
                               val_f1[0].mean(), val_f1[1],
                               valid_mAP, valid_ap
                              ]
        
        """wandb visualization"""
        if not args.val:
            wandb.log({k: v for k, v in zip(total_results_index, total_results_value)})
        else:
            wandb.log({k: v for k, v in zip(total_results_index[2:], total_results_value[2:])})

        """ save checkpoint """
        if val_cls_acc > best_score:
            best_score = val_cls_acc
            print('\nSaving state, epoch : %d , Validation accuracy : %.4f \n' % (current_epoch, val_cls_acc))
            state = {
                'model': model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                'epoch': current_epoch,
                'iter': args.global_counter,
            }
            model_file = os.path.join(args.save_folder, 'best.pth')
            torch.save(state, model_file)

        else:
            print(f'\nStill best Validation accuracy is {best_score:.4f}\n')