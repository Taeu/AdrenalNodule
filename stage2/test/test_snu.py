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
    # exp
    args.add_argument("--pkl_name", type=str, default='det2_set1_epoch_82_0.pkl')
    args.add_argument("--exp_name", type=str, default='unet_1')  # det2_set1_dc_and_fc_with_input_160
    args.add_argument("--model_name", type=str, default='unet')
    args.add_argument("--cuda", type=bool, default=True)

    # dataset
    args.add_argument("--data_dir", type=str, default='./ag/data/Numpy2/')
    args.add_argument("--pkl_dir", type=str, default='./ag/result/1024/')
    args.add_argument("--checkpoint_dir", type=str,
                      default='./ag/result/stage2_0916/checkpoints/')
    args.add_argument("--image_size", type=int, default=160)
    args.add_argument("--gpu_id", type=str, default="0")
    args.add_argument("--save_result", action='store_true', default=False)
    args.add_argument('--class_head', action='store_true', default=False)
    config = args.parse_args()
    return config


# snu
# python test_snu.py --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 0 --save_result
# python test_snu.py --pkl_name det2_set1_epoch_82_250.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 0 --save_result
# python test_snu.py --pkl_name det2_set1_epoch_82_500.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 0 --save_result
# python test_snu.py --pkl_name det2_set1_epoch_82_750.pkl --exp_name det2_set1_dc_and_fc_with_input_160 --gpu_id 0 --save_result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def windowing(im, win=[-350, 350]):
    """scale intensity from win[0]~win[1] to float numbers in 0~255"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

def get_dice(gt, seg):
    dice = np.sum(seg[gt==True])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice

def get_output(image, output, cur_gt, rp, side, fn_pred_dict, row, class_head=False, args=None):
    # get 160 patch from output slice
    for idx, r in enumerate(rp):
        cx, cy = r.centroid
        cx, cy = int(cx), int(cy)
        x1, x2, y1, y2 = max(0, cx - side), min(512, cx + side), max(0, cy - side), min(512, cy + side)
        h, w = x2 - x1, y2 - y1
        crop_image = image[x1:x2, y1:y2]
        crop_output = output[x1:x2, y1:y2]
        crop_gt = cur_gt[x1:x2, y1:y2]
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
        input = windowing(input)
        input = input.astype(np.uint8)
        input = tfs_val(input)
        input = input.unsqueeze(0)
        with torch.no_grad():
            input = input.cuda()
            if class_head:
                _, pred_label = model(input)
            else:
                pred_label = model(input)
            cur_pred_label = pred_label.cpu().numpy()

        if np.sum(crop_gt) == 0 and np.sum(sigmoid(cur_pred_label) >= 0.5) == 0:
            return -1, -1, -1
        #import pdb; pdb.set_trace()
        dice = get_dice(crop_gt, sigmoid(cur_pred_label).squeeze() >= 0.5)
        #print(dice, np.sum(crop_gt), np.sum(sigmoid(cur_pred_label).squeeze() >= 0.5))
        return dice, sigmoid(cur_pred_label).squeeze() >= 0.5, crop_gt
        #
        # if row.fns in fn_pred_dict:
        #     if class_head:
        #         fn_pred_dict[f'{row.fns}_{row.idx}'].append(cur_pred_label)  # max(cur_pred_label, fn_max_dict[row.fns])
        #     elif args.save_result:
        #         fn_pred_dict[f'{row.fns}_{row.idx}_output'].append(cur_pred_label)
        #         fn_pred_dict[f'{row.fns}_{row.idx}_input'].append(input.cpu().numpy())
        #     else:
        #         fn_pred_dict[f'{row.fns}_{row.idx}'].append(np.max(cur_pred_label))
        # else:
        #     if class_head:
        #         fn_pred_dict[f'{row.fns}_{row.idx}'] = [cur_pred_label]
        #     elif args.save_result:
        #         fn_pred_dict[f'{row.fns}_{row.idx}_output'] = [cur_pred_label]
        #         fn_pred_dict[f'{row.fns}_{row.idx}_input'] = [input.cpu().numpy()]
        #     else:
        #         fn_pred_dict[f'{row.fns}_{row.idx}'] = [np.max(cur_pred_label)]

        break
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

def get_stage1_output():
    root_dir = './ag/result/'
    hos = '1024/'  # GC : 1103_gc, SC : 220223_s
    df = pd.DataFrame()
    for i in tqdm([0, 250, 500, 750]):
        fname = f'det2_set1_epoch_82_{i}.pkl'
        with open(root_dir + hos + fname, 'rb') as f:
            cur_df = pickle.load(f)
            cur_df = cur_df[cur_df['tvt'] == 3]
        # print(len(cur_df))
        df = pd.concat([df, cur_df], ignore_index=True)
    print(len(df), len(set(df.fns.tolist())))
    return df

if __name__ == '__main__':
    args = make_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # For gpu

    if 'unet' in args.model_name:
        model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )

    checkpoint_dir = './ag/result/stage2_0916/checkpoints/det2_set1_dc_and_fc_with_input_160/best_epoch.pth'
    args.exp_name = 'det2_set1_dc_and_fc_with_input_160'
    if args.class_head:
        model = UNet_(model, 1)
        checkpoint_dir = './ag/result/stage2_0916/checkpoints/1224_det2_set1_dc_and_fc_with_input_160/best_epoch.pth'
        args.exp_name = '1224_det2_set1_dc_and_fc_with_input_160'
    # checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name, 'best_epoch.pth')

    model = model.cuda()
    model.load_state_dict(torch.load(checkpoint_dir)['model'])
    model.eval()

    if 'focal' in args.exp_name:
        criterion = DiceLoss()
    elif 'dice' in args.exp_name:
        criterion = DiceLoss()
    else:
        criterion = DiceLoss()

    args.save_dir = args.pkl_dir
    criterion = criterion.cuda() if args.cuda else criterion
    if args.save_result:
        os.makedirs(args.save_dir + args.exp_name, exist_ok=True)

    fn_pred_dict_0 = {}
    fn_pred_dict_1 = {}
    epoch_loss = []

    ## stage1 output pickle load
    with open(args.pkl_dir + args.pkl_name, 'rb') as f:
        df = pickle.load(f)
        df = df[df.tvt == 3]



    df = get_stage1_output()
    ###
    # df = pd.read_pickle(args.pkl_dir + args.pkl_name) # './ag/code/mulan_eval/results/GC_STAGE1_det2.pkl'
    test_data_dir = args.data_dir  # './ag/data/gc_npy/'

    side = 80
    left_dice = {}
    right_dice = {}
    for index, row in tqdm(df.iterrows()):
        # cols : fns,idx,bbox,score,label
        if len(row.bbox) == 0:
            continue
        cur_slice = np.zeros((2, 512, 512))
        for b in row.bbox:
            y_middle = (b[1] + b[3]) / 2
            idx = 0
            if y_middle > 256:
                idx = 1
            cur_slice[idx, int(b[0]):int(b[2]), int(b[1]): int(b[3])] = 1
        cur_slice = cur_slice.astype(np.uint8)
        if 'Numpy2' in args.data_dir or 'numpy' in args.data_dir:
            # shape : H,W,C to H,W
            image = np.load(test_data_dir + row.fns)[:, :, row.idx]
            gt = np.load('./ag/data/Label_npy/' + row.fns)[:,:,row.idx]
        else:
            # shape : C,H,W to H,W
            image = np.load(test_data_dir + row.fns)[row.idx]

        # right
        output = cur_slice[0]
        cur_gt = gt == 3
        rp = regionprops(output)
        if len(rp) == 0:
            if np.sum(cur_gt) != 0 :
                dice = 0
        else:
            dice, output, label = get_output(image, output,cur_gt, rp, side, fn_pred_dict_0, row, args.class_head, args)
        if dice != -1 and not (len(rp) == 0 and np.sum(cur_gt) == 0):
            right_dice[row.fns + str(row.idx)] = [dice, output, label]
            #print(f'right {row.fns}, {row.idx}, dice : {dice}')
        # left
        cur_gt = gt == 4
        output = cur_slice[1]
        rp = regionprops(output)
        if len(rp) == 0:
            if np.sum(cur_gt) != 0:
                dice = 0
        else:
            dice, output, label  = get_output(image, output,cur_gt, rp, side, fn_pred_dict_1, row, args.class_head, args)
        if dice != -1 and not (len(rp) == 0 and np.sum(cur_gt) == 0):
            left_dice[row.fns + str(row.idx)] = [dice, output, label]
            #print(f'left {row.fns}, {row.idx}, dice : {dice}')

    with open('./ag/result/220316/left_dice.pkl', 'wb') as f:
        pickle.dump(left_dice, f)
    with open('./ag/result/220316/right_dice.pkl', 'wb') as f:
        pickle.dump(right_dice, f)
    if args.class_head:
        args.exp_name + '_class_head'

    # is_result_save = '0202_'
    # if args.save_result:
    #     is_result_save += 'save_result_'
    # save_0_path = os.path.join(args.save_dir, is_result_save + args.exp_name + '_0_' + args.pkl_name)
    # with open(save_0_path, 'wb') as f:
    #     pickle.dump(fn_pred_dict_0, f)
    #
    # save_1_path = os.path.join(args.save_dir, is_result_save + args.exp_name + '_1_' + args.pkl_name)
    # with open(save_1_path, 'wb') as f:
    #     pickle.dump(fn_pred_dict_1, f)
