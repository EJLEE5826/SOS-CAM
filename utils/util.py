import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image 
import wandb 

from utils.decode import decode_seg_map_sequence


def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()


def load_img_label_list_from_npy(img_name_list, dataset):
    cls_labels_dict = np.load(f'/home/junehyoung/code/wsss_baseline/voc2012_list/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]

def output_visualize(image, cam, agg_cam, label, pred_map, agg_pred_map):

    def cam2img(input):
        input *= 255
        input = np.clip(input, 0, 255)
        input = input.squeeze(0)
        input = cv2.applyColorMap(input.astype(np.uint8), cv2.COLORMAP_JET)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        return np.float32(input)
    
    image = np.transpose(image.clone().cpu().detach().numpy(), (1,2,0))  # H, W, C
    
    """ image denormalize """
    image *= [0.229, 0.224, 0.225]
    image += [0.485, 0.456, 0.406]
    image *= 255
    image = np.clip(image.transpose(2,0,1), 0, 255).astype(np.uint8) # C, H, W

    size = image.shape[1]

    """ visualize selected CAM outputs """
    label = label.clone().cpu().detach().numpy()
    nonzero_label = np.nonzero(label)[0]

    num_select_imgs = len(label) + 5

    selected_cam_image = np.zeros((num_select_imgs, 3, size, size), dtype=np.uint8) # ((image, cam1, cam2, .., pseudo, gt), 3, 320, 320)
    if selected_cam_image[0].shape != image.shape:
        print(f"{selected_cam_image[0].shape}, {image.shape}")
    selected_cam_image[0] = image
    
    cam_img = cam2img(cam)
    selected_cam_image[len(label)+1] = cam_img.transpose(2, 0, 1)

    selected_cam_image[len(label)+2] = decode_seg_map_sequence(pred_map) * 255

    if agg_cam is not None:
        agg_cam_img = cam2img(agg_cam)
        selected_cam_image[-2] = agg_cam_img.transpose(2, 0, 1)

    """ visualize semantic segmentaiton map """
    if agg_pred_map is not None:
        selected_cam_image[-1] = decode_seg_map_sequence(agg_pred_map) * 255
        
    selected_cam_image = selected_cam_image.astype(np.float32) / 255.
        
    return selected_cam_image, nonzero_label