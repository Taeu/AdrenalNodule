
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
    args.add_argument("--exp_name", type=str, default = 'det2_unet3_dc_and_fc') # det2_set1_dc_and_fc_with_input_160
    args.add_argument("--model_name", type=str, default='unet')
    args.add_argument("--cuda", type=bool, default=True)

    #dataset
    args.add_argument("--data_dir", type=str, default='.//ag/data/Numpy2/')
    args.add_argument("--pkl_dir", type=str, default='./ag/result/1024/')
    args.add_argument("--checkpoint_dir", type=str, default='./ag/result/stage2_220626/checkpoints/')
    args.add_argument("--num_classes", type=int, default=3)
    args.add_argument("--image_size", type=int, default=160)
    args.add_argument("--gpu_id", type=str, default="0")
    args.add_argument("--save_result", action='store_true', default=False)
    args.add_argument('--class_head', action='store_true', default=False)
    args.add_argument('--squeeze', action='store_true', default=False)

    config = args.parse_args()
    return config

# snu
# python test_gc3.py --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 0 --save_result
# python test_gc3.py --pkl_name det2_set1_epoch_82_250.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 2 --save_result
# python test_gc3.py --pkl_name det2_set1_epoch_82_500.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 2 --save_result
# python test_gc3.py --pkl_name det2_set1_epoch_82_750.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 2 --save_result

# gc
# python test_gc3.py --data_dir .//ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
#  --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 1 --save_result
# python test_gc3.py --data_dir .//ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
# --pkl_name det2_set1_epoch_82_1000.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 1 --save_result

# python test_gc3.py --data_dir .//ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
#  --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 1 --class_head
# python test_gc3.py --data_dir .//ag/data/gc_npy/  --pkl_dir ./ag/result/1103_gc/ \
# --pkl_name det2_set1_epoch_82_1000.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 0 --class_head


# sc
# python test_gc3.py --data_dir .//ag/data2/numpy/ --pkl_dir ./ag/result/1103_s/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 1
# python test_gc3.py --data_dir .//ag/data2/numpy/ --pkl_dir ./ag/result/1103_s/ \
# --pkl_name det2_set1_epoch_82_600.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 0

# new 220223
# python test_gc3.py --data_dir .//ag/data3/numpy/ --pkl_dir ./ag/result/220223_s/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 2


# new snu 220502
# python test_gc3.py --data_dir .//ag/data4/numpy/ --pkl_dir ./ag/result/220428_snu/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 2 --save_result


# python test_gc3.py --data_dir .//ag/data2/numpy/ --pkl_dir ./ag/result/1103_s/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 1 --class_head
# python test_gc3.py --data_dir ./ag/data2/numpy/ --pkl_dir ./ag/result/1103_s/ \
# --pkl_name det2_set1_epoch_82_600.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 0 --class_head

# new_gc 220518
# python test_gc3.py --data_dir .//ag/data5/numpy/ --pkl_dir ./ag/result/220516_gc/ \
# --pkl_name det2_set1_epoch_82_0.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 0 --save_result

# python test_gc3.py --data_dir .//ag/data5/numpy/ --pkl_dir ./ag/result/220516_gc/ \
# --pkl_name det2_set1_epoch_82_3000.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 0 --save_result

# python test_gc3.py --data_dir .//ag/data5/numpy/ --pkl_dir ./ag/result/220516_gc/ \
# --pkl_name det2_set1_epoch_82_6000.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 1 --save_result

# python test_gc3.py --data_dir .//ag/data5/numpy/ --pkl_dir ./ag/result/220516_gc/ \
# --pkl_name det2_set1_epoch_82_9000.pkl --exp_name det2_unet3_dc_and_fc --gpu_id 3 --save_result

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

def get_pred(image, output, fn_pred_dict, row, class_head = False, args = None):
    # get 160 patch from output slice

        image = image.astype(np.uint8)
        output = output.astype(np.uint8)

        new_img = tfs_val(image)
        new_output = tfs_val(output)
        new_img = torch.cat((new_img, new_output), 0)
        new_img = new_img.unsqueeze(0)

        with torch.no_grad():
            input = new_img.cuda()

            if 'hrnet' in args.exp_name:
                pred_label = model(input)[-1]
            else:
                pred_label = model(input)
            cur_pred_label = pred_label[:,1:2] # .cpu().numpy() # (1, 3, 160, 160) -> (1, 1, 160, 160)

        if 'hrnet' not in args.exp_name:
            cur_pred_label = torch.nn.functional.interpolate(cur_pred_label, size=(40, 40), mode='bilinear')
        cur_pred_label = cur_pred_label.cpu().detach().numpy()
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
from models.seg_hrnet_ocr import *
from config import config
from config import update_config
import yaml
up4 = nn.Upsample(scale_factor=4, mode='nearest')


def get_output(df, index, row, slice_num=3):
    cur_slice = np.zeros((slice_num * 2, 512, 512))  # left slice_num , right slice_num

    for b in row.bbox:
        update_output_slice(cur_slice, b, slice_num, add_idx=0)

    if slice_num == 3:
        row_before = df.iloc[index - 1]
        if index + 1 == len(df) :
            row_after = df.iloc[0]
        else:
            row_after = df.iloc[index + 1]
        # left
        if row_before.fns == row.fns and row_before.idx + 1 == row.idx:
            for b in row_before.bbox:
                update_output_slice(cur_slice, b, slice_num, add_idx=-1)
        if row_after.fns == row.fns and row_after.idx - 1 == row.idx:
            for b in row_after.bbox:
                update_output_slice(cur_slice, b, slice_num, add_idx=+1)

    return cur_slice


def update_output_slice(cur_slice, b, slice_num, add_idx=0):
    y_middle = (b[1] + b[3]) / 2
    idx = slice_num // 2 + add_idx  # 0
    if y_middle > 256:

        idx = (slice_num * 2) // 2 + 1 + add_idx
        # if idx == 4:
        #     import pdb; pdb.set_trace()

    cur_slice[idx, int(b[0]):int(b[2]), int(b[1]): int(b[3])] = 1


def crop_region(image, label, output, slice_num, fn, is_left=True):

    if is_left:
        idx = slice_num // 2 # 1
        save_num = '0'
    else:
        idx = (slice_num * 2) // 2 + 1 ##
        save_num = '1'
        #import pdb; pdb.set_trace()
    cur_output = output[idx]
    rp = regionprops(cur_output)
    if len(rp) == 0:
        return None, None
    else:
        for idx, r in enumerate(rp):
            cx, cy = r.centroid
            cx, cy = int(cx), int(cy)
            x1, x2, y1, y2 = max(0, cx - side), min(512, cx + side), max(0, cy - side), min(512, cy + side)
            h, w = x2 - x1, y2 - y1
            crop_image = image[x1:x2, y1:y2]
            if label is not None:
                crop_label = label[x1:x2, y1:y2]
                bb = np.zeros((160, 160, slice_num))
                bb[:crop_image.shape[0], :crop_image.shape[1]] = crop_label
                crop_label = bb.astype(np.uint8)

            if is_left:
                crop_output = output[:3, x1:x2, y1:y2]
            else:
                crop_output = output[3:, x1:x2, y1:y2]
            crop_output = crop_output.transpose((1, 2, 0))

            aa = np.zeros((160, 160, slice_num))

            cc = np.zeros((160, 160, slice_num))
            aa[:crop_image.shape[0], :crop_image.shape[1]] = crop_image
            cc[:crop_image.shape[0], :crop_image.shape[1]] = crop_output * crop_image
            crop_image = aa.astype(np.int16)
            crop_output = cc.astype(np.int16)

            # print(crop_output.shape, (side*2, side*2))
            # if crop_output.shape != (side * 2, side * 2, slice_num) or crop_label.shape != (
            # side * 2, side * 2, slice_num) or crop_image.shape != (side * 2, side * 2, slice_num):
            #     print(crop_output.shape, h, w, 160 - h, 160 - w)
            # assert crop_output.shape == (side * 2, side * 2, slice_num) or crop_label.shape == (
            # side * 2, side * 2, slice_num) or crop_image.shape == (side * 2, side * 2, slice_num)
            return crop_image, crop_output

if __name__ == '__main__':
    args = make_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # For gpu

    if 'unet' in args.exp_name or args.exp_name == 'det2_set1_dc_and_fc_with_input_160':
        encoder_name = 'resnet34'
        if args.squeeze:
            encoder_name = 'se_resnet50'
        model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3 * 2,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,  # model output channels (number of classes in your dataset)
        )

    elif 'psp' in args.exp_name:
        encoder_name = 'resnet34'
        if args.squeeze:
            encoder_name = 'se_resnet50'
        model = smp.PSPNet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3 * 2,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,  # model output channels (number of classes in your dataset)
        )
    elif 'hrnet' in args.exp_name:

        args.cfg = 'cfg_hrnet.yaml'
        args.opts = []
        # with open("cfg_hrnet.yaml", "r") as f:
        #     args.cfg = yaml.load(f)
        update_config(config, args)
        model = HighResolutionNet(config)
        model.init_weights(config.MODEL.PRETRAINED)

    checkpoint_dir = f'{args.checkpoint_dir}{args.exp_name}/best_epoch.pth'
    #args.exp_name = 'det2_set1_dc_and_fc_with_input_160'
    if args.class_head:
        model = UNet_(model, 1)
        checkpoint_dir = './ag/result/stage2_0916/checkpoints/1224_det2_set1_dc_and_fc_with_input_160/best_epoch.pth'
        args.exp_name = '1224_det2_set1_dc_and_fc_with_input_160'
    #checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name, 'best_epoch.pth')

    model = model.cuda()
    model.load_state_dict(torch.load(checkpoint_dir)['model'])
    model.eval()


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
    #df = pd.read_pickle(args.pkl_dir + args.pkl_name) # './ag/code/mulan_eval/results/GC_STAGE1_det2.pkl'
    test_data_dir = args.data_dir #'./ag/data/gc_npy/'

    side = 80
    slice_num = 3
    add_num = 1
    print(f'total inference files : {len(df)}')
    # todo : debug_preprocess + dataloader.py 의 data_loader3 과 기존 inference code 를 합쳐서 만들 것.
    for index, row in tqdm(df.iterrows()):

        # cols : fns,idx,bbox,score,label
        if len(row.bbox) == 0:
            continue

        output_slice = get_output(df, index, row, slice_num=slice_num)
        output_slice = output_slice.astype(np.uint8)
        if ('Numpy2' in args.data_dir or 'numpy' in args.data_dir) and 'gc_npy' not in args.data_dir:
            # shape : H,W,C to H,W
            image = windowing(np.load(args.data_dir + row.fns)[:, :, row.idx - add_num: row.idx + 1 + add_num])
            #label = np.load(args.label_dir + row.fns)[:, :, row.idx - add_num: row.idx + 1 + add_num]
        else:
            # shape : C,H,W to H,W
            image = windowing(np.load(args.data_dir + row.fns)[row.idx - add_num: row.idx + 1 + add_num])
            image = image.transpose((1,2,0))
            #label = np.load(args.label_dir + row.fns)[:, :, row.idx - add_num: row.idx + 1 + add_num]
        image = image.astype(np.uint8)
        #label = label.astype(np.uint8)
        label = None
        try:
            image, output = crop_region(image, label, output_slice, slice_num, f'{row.fns[:-4]}_{row.idx}', is_left=True)
            if image is not None and output is not None:
                get_pred(image, output, fn_pred_dict_0, row, args.class_head, args)
        except Exception as e :
            print(row.fns, row.idx, e)
            pass

        output_slice = get_output(df, index, row, slice_num=slice_num)
        output_slice = output_slice.astype(np.uint8)
        if ('Numpy2' in args.data_dir or 'numpy' in args.data_dir) and 'gc_npy' not in args.data_dir:
            # shape : H,W,C to H,W
            image = windowing(np.load(args.data_dir + row.fns)[:, :, row.idx - add_num: row.idx + 1 + add_num])
            # label = np.load(args.label_dir + row.fns)[:, :, row.idx - add_num: row.idx + 1 + add_num]
        else:
            # shape : C,H,W to H,W
            image = windowing(np.load(args.data_dir + row.fns)[row.idx - add_num: row.idx + 1 + add_num])
            image = image.transpose((1, 2, 0))
            # label = np.load(args.label_dir + row.fns)[:, :, row.idx - add_num: row.idx + 1 + add_num]
        image = image.astype(np.uint8)
        try:
            image, output = crop_region(image, label, output_slice, slice_num, f'{row.fns[:-4]}_{row.idx}', is_left=False)
            if image is not None and output is not None:
                get_pred(image, output, fn_pred_dict_1, row, args.class_head, args)
        except Exception as e :
            print(row.fns, row.idx, e)
            pass
        #import pdb; pdb.set_trace()
    is_result_save = '220626_unet3_' ## todo : auto
    if args.save_result:
        is_result_save += 'save_result_'
    save_0_path = os.path.join(args.save_dir, is_result_save + args.exp_name + '_0_' + args.pkl_name)
    with open(save_0_path,'wb') as f:
        pickle.dump(fn_pred_dict_0, f)

    save_1_path = os.path.join(args.save_dir,is_result_save +args.exp_name + '_1_' + args.pkl_name)
    with open(save_1_path,'wb') as f:

        pickle.dump(fn_pred_dict_1, f)
    #import pdb; pdb.set_trace()