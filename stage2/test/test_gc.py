
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
from dataloader import data_loader
import pickle5 as pickle
# model
import segmentation_models_pytorch as smp
from networks import UNet_

# losses
from losses import DiceLoss, FocalLoss
from skimage.measure import regionprops
import pandas as pd
from torchvision import datasets, transforms




tfs_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])

def make_parser():
    args = argparse.ArgumentParser()
    #exp
    args.add_argument("--pkl_name", type=str, default='det2_set1_epoch_82_0.pkl')
    args.add_argument("--exp_name", type=str, default = 'unet_1') # det2_set1_dc_and_fc_with_input_160
    args.add_argument("--model_name", type=str, default='unet')
    args.add_argument("--cuda", type=bool, default=True)

    #dataset
    args.add_argument("--data_dir", type=str, default='./ag/data/Numpy2/')
    args.add_argument("--pkl_dir", type=str, default='./ag/result/1024/')
    args.add_argument("--save_dir", type=str, default='./ag/result/stage2/')
    args.add_argument("--checkpoint_dir", type=str, default='./ag/result/stage2_0916/checkpoints/')
    args.add_argument("--image_size", type=int, default=160)
    
    args.add_argument("--start_idx", type=int, default=0)
    args.add_argument("--end_idx", type=int, default=16000)
    
    args.add_argument("--gpu_id", type=str, default="0")
    args.add_argument("--save_result", action='store_true', default=False)
    args.add_argument('--class_head', action='store_true', default=False)
    args.add_argument('--squeeze', action='store_true', default=False)

    config = args.parse_args()
    return config

# snu
# python test_gc.py --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2 --save_result
# python test_gc.py --pkl_name det2_set1_epoch_82_250.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2 --save_result
# python test_gc.py --pkl_name det2_set1_epoch_82_500.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2 --save_result
# python test_gc.py --pkl_name det2_set1_epoch_82_750.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2 --save_result

# gc
# python test_gc.py --data_dir ./ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
#  --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 1 --save_result
# python test_gc.py --data_dir ./ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
# --pkl_name det2_set1_epoch_82_1000.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 1 --save_result

# python test_gc.py --data_dir ./ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
#  --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 1 --class_head
# python test_gc.py --data_dir ./ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
# --pkl_name det2_set1_epoch_82_1000.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 0 --class_head


# sc
# python test_gc.py --data_dir ./ag/data2/numpy/ --pkl_dir ./ag/result/1103_s/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 1
# python test_gc.py --data_dir ./ag/data2/numpy/ --pkl_dir ./ag/result/1103_s/ \
# --pkl_name det2_set1_epoch_82_600.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 0

# new 220223
# python test_gc.py --data_dir ./ag/data3/numpy/ --pkl_dir ./ag/result/220223_s/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2


# new snu 220502
# python test_gc.py --data_dir ./ag/data4/numpy/ --pkl_dir ./ag/result/220428_snu/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 2 --save_result


# python test_gc.py --data_dir ./ag/data2/numpy/ --pkl_dir ./ag/result/1103_s/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 1 --class_head
# python test_gc.py --data_dir ./ag/data2/numpy/ --pkl_dir ./ag/result/1103_s/ \
# --pkl_name det2_set1_epoch_82_600.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 0 --class_head

# new_gc 220518
# python test_gc.py --data_dir ./ag/data5/numpy/ --pkl_dir ./ag/result/220516_gc/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 0 --save_result

# python test_gc.py --data_dir ./ag/data5/numpy/ --pkl_dir ./ag/result/220516_gc/ \
# --pkl_name det2_set1_epoch_82_3000.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 0 --save_result

# python test_gc.py --data_dir ./ag/data5/numpy/ --pkl_dir ./ag/result/220516_gc/ \
# --pkl_name det2_set1_epoch_82_6000.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 1 --save_result

# python test_gc.py --data_dir ./ag/data5/numpy/ --pkl_dir ./ag/result/220516_gc/ \
# --pkl_name det2_set1_epoch_82_9000.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 3 --save_result

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def windowing(im, win = [-350, 350]):
    """scale intensity from win[0]~win[1] to float numbers in 0~255"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

def get_output(image, output, rp, side, fn_pred_dict, row, class_head = False, args = None):
    # get 160 patch from output slice
    for idx, r in enumerate(rp):
        cx, cy = r.centroid
        cx, cy = int(cx), int(cy)
        x1, x2, y1, y2 = max(0, cx - side), min(512, cx + side), max(0, cy - side), min(512, cy + side)
        h, w = x2 - x1, y2 - y1
        crop_image = image[x1:x2, y1:y2]
        crop_output = output[x1:x2, y1:y2]

        aa = np.zeros((160, 160))
        cc = np.zeros((160, 160))
        aa[:crop_image.shape[0], :crop_image.shape[1]] = crop_image
        cc[:crop_image.shape[0], :crop_image.shape[1]] = crop_output
        crop_image = aa.astype(np.int16)
        crop_output = cc.astype(np.uint8)

        # inference
        # crop_image : 160, 160, crop_output 160, 160
        crop_image_output = crop_image * crop_output
        input = np.zeros((160, 160, 3))
        input[:, :, 0] = crop_image
        input[:, :, 1] = crop_image_output
        input[:, :, 2] = crop_image_output
        original_input = input.copy()
        input = windowing(input)
        input = input.astype(np.uint8)
        input = tfs_val(input)
        input = input.unsqueeze(0)
        with torch.no_grad():
            input = input.cuda()
            if class_head:
                _, pred_label = model(input)
            else:
                if 'hrnet' in args.exp_name:
                    pred_label = model(input)[-1]
                else:
                    pred_label = model(input)
            cur_pred_label = pred_label # .cpu().numpy()

        if 'hrnet' not in args.exp_name:
            cur_pred_label = torch.nn.functional.interpolate(cur_pred_label, size=(40, 40), mode='bilinear')
        cur_pred_label = cur_pred_label.cpu().numpy()
        #import pdb; pdb.set_trace()
        if row.fns in fn_pred_dict:
            if class_head :
                fn_pred_dict[f'{row.fns}_{row.idx}'].append(cur_pred_label) #max(cur_pred_label, fn_max_dict[row.fns])
            elif args.save_result:
                fn_pred_dict[f'{row.fns}_{row.idx}_output'].append(cur_pred_label.astype(np.float16))
                #fn_pred_dict[f'{row.fns}_{row.idx}_input'].append(original_input.astype(np.int16))
            else:
                fn_pred_dict[f'{row.fns}_{row.idx}'].append(np.max(cur_pred_label))
        else:
            if class_head :
                fn_pred_dict[f'{row.fns}_{row.idx}'] = [cur_pred_label]
            elif args.save_result:
                fn_pred_dict[f'{row.fns}_{row.idx}_output'] = [cur_pred_label.astype(np.float16)]
                #fn_pred_dict[f'{row.fns}_{row.idx}_input'] = [original_input.astype(np.int16)]
            else :
                fn_pred_dict[f'{row.fns}_{row.idx}'] = [np.max(cur_pred_label)]


        # vis
        # if args.save_result:
        #     if np.sum(sigmoid(cur_pred[i]) > 0.5) != 0 or np.sum(label[i].cpu().numpy()) != 0:
        #         save_fn = os.path.join(save_root, args.exp_name, fns[i].replace('.npy', '') + '.png')
        #         draw_npy = np.zeros((args.image_size, args.image_size * 4))
        #         draw_npy[:, :args.image_size] = (image[i][0].cpu().numpy() * 0.229 + 0.485) * 255
        #         draw_npy[:, args.image_size: args.image_size * 2] = (image[i][
        #                                                                  0].cpu().numpy() * 0.229 + 0.485) * 255 * 0 + \
        #                                                             label[i].cpu().numpy() * 255 * 1
        #         draw_npy[:, args.image_size * 2: args.image_size * 3] = (image[i][
        #                                                                      1].cpu().numpy() * 0.224 + 0.456) * 255
        #         draw_npy[:, args.image_size * 3:] = (image[i][0].cpu().numpy() * 0.229 + 0.485) * 255 * 0 + (
        #             sigmoid(cur_pred[i])) * 255 * 1
        #
        #         draw_npy = draw_npy.astype(np.uint8)
        #         cv2.imwrite(save_fn, draw_npy)

# for HRNET-OCR
from config import config
from config import update_config
import yaml

if __name__ == '__main__':
    args = make_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # For gpu

    if 'unet' in args.exp_name or args.exp_name == 'det2_set1_dc_and_fc_with_input_160':
        encoder_name = 'resnet34'
        model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )

    checkpoint_dir = f'./ag/result/stage2_0916/checkpoints/{args.exp_name}/best_epoch.pth'
    checkpoint_dir = f'./ag/result/stage2/checkpoints/{args.exp_name}/best_epoch.pth'
    model = model.cuda()
    model.load_state_dict(torch.load(checkpoint_dir)['model'])
    model.eval()

    if 'focal' in args.exp_name:
        criterion = DiceLoss()
    elif 'dice' in args.exp_name :
        criterion = DiceLoss()
    else:
        criterion = DiceLoss()

    args.save_dir = args.pkl_dir
    criterion = criterion.cuda() if args.cuda else criterion
    if args.save_result:
        os.makedirs(args.save_dir + args.exp_name , exist_ok=True)

    fn_pred_dict_0 = {}
    fn_pred_dict_1 = {}
    epoch_loss = []

    with open(args.pkl_dir + args.pkl_name,'rb') as f:
        df = pickle.load(f)
    
    test_data_dir = args.data_dir #'./ag/data/gc_npy/'

    side = 80
    for index, row in tqdm(df.iterrows()):
        # cols : fns,idx,bbox,score,label
        if len(row.bbox) == 0:
            continue
        cur_slice = np.zeros((2,512,512))
        for b in row.bbox:
            y_middle = (b[1] + b[3])/2
            idx = 0
            if y_middle > 256:
                idx = 1
            if (b[3] - b[1]) > 120 and b[3] > 256 and b[1] < 256 and (b[2] - b[0]) < 60: ## added for post-processing cases such as aorta
                continue
            cur_slice[idx, int(b[0]):int(b[2]), int(b[1]): int(b[3])] = 1
        cur_slice = cur_slice.astype(np.uint8)
        if 'Numpy2' in args.data_dir or 'numpy' in args.data_dir:
            # shape : H,W,C to H,W
            image = np.load(test_data_dir + row.fns)[:,:,row.idx]
        else:
            # shape : C,H,W to H,W
            image = np.load(test_data_dir + row.fns)[row.idx]

        # right
        output = cur_slice[0]
        rp = regionprops(output)
        if len(rp) == 0:
            pass
        else:
            get_output(image, output, rp, side, fn_pred_dict_0, row, args.class_head, args)

        # left
        output = cur_slice[1]
        rp = regionprops(output)
        if len(rp) == 0:
            pass
        else:
            get_output(image, output, rp, side, fn_pred_dict_1, row, args.class_head, args)
    if args.class_head:
        args.exp_name +'_class_head'

    is_result_save = '220627_aorta_' ## todo : auto
    if args.save_result:
        is_result_save += 'save_result_'
    save_0_path = os.path.join(args.save_dir, is_result_save + args.exp_name + '_0_' + args.pkl_name)
    with open(save_0_path,'wb') as f:
        pickle.dump(fn_pred_dict_0, f)

    save_1_path = os.path.join(args.save_dir,is_result_save +args.exp_name + '_1_' + args.pkl_name)
    with open(save_1_path,'wb') as f:
        pickle.dump(fn_pred_dict_1, f)
