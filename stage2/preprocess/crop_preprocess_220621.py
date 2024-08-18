import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from skimage.measure import regionprops
import os.path as osp
from tqdm import tqdm
import argparse

def make_parser():
    args = argparse.ArgumentParser()

    #exp
    args.add_argument("--data_dir", type=str, default='./ag/data/stage2/')
    args.add_argument("--input_dir", type=str, default='stage1_input_unet')
    args.add_argument("--label_dir", type=str, default='stage1_label_train_unet')
    args.add_argument("--output_dir", type=str, default='stage1_output_train_unet')
    args.add_argument("--version", type=str, default='_det4') # _det2

    config = args.parse_args()
    return config
# python crop_preprocess.py --input_dir stage1_input_train --label_dir stage1_label_train \
# --output_dir stage1_output_train_det2 --version _det2

if __name__ == '__main__':
    args = make_parser()
    data_dir = args.data_dir
    s1_input_path = data_dir + args.input_dir
    s1_label_path = data_dir + args.label_dir
    s1_output_path = data_dir + args.output_dir

    inputs = os.listdir(s1_input_path)
    labels = os.listdir(s1_label_path)
    outputs = os.listdir(s1_output_path)

    outputs.sort()

    ###############
    # output left, right 나눠서 작업
    # SNU_0860_142.npy 이런식으로 저장되어있음.
    # output left 는 _0, right _1 로 또 구분해주기
    ##############

    to_input_dir = s1_input_path + args.version + '_crop/'
    to_label_dir = s1_label_path + args.version + '_crop/'
    to_output_dir = s1_output_path + '_crop/'
    os.makedirs(to_input_dir, exist_ok=True)
    os.makedirs(to_label_dir, exist_ok=True)
    os.makedirs(to_output_dir, exist_ok=True)

    side = 80
    for fn in tqdm(outputs):
        image = np.load(osp.join(s1_input_path, fn))
        label = np.load(osp.join(s1_label_path, fn))
        output = np.load(osp.join(s1_output_path, fn))

        # left
        output0 = output[0]
        rp0 = regionprops(output0)
        if len(rp0) == 0:
            pass
        else:
            for idx, r in enumerate(rp0):
                cx,cy = r.centroid
                cx,cy = int(cx), int(cy)
                x1, x2, y1, y2 = max(0, cx-side), min(512, cx+side), max(0,cy-side), min(512, cy+side)
                h, w = x2-x1, y2-y1
                crop_image = image[x1:x2,y1:y2]
                crop_label = label[x1:x2,y1:y2]
                crop_output = output0[x1:x2,y1:y2]

                aa = np.zeros((160,160))
                bb = np.zeros((160,160))
                cc=  np.zeros((160,160))
                aa[:crop_image.shape[0], :crop_image.shape[1]] = crop_image
                bb[:crop_image.shape[0], :crop_image.shape[1]] = crop_label
                cc[:crop_image.shape[0], :crop_image.shape[1]] = crop_output
                crop_image = aa.astype(np.int16)
                crop_label = bb.astype(np.uint8)
                crop_output = cc.astype(np.uint8)

                #print(crop_output.shape, (side*2, side*2))
                if crop_output.shape != (side*2, side*2) or crop_label.shape != (side * 2, side*2) or crop_image.shape != (side * 2, side*2):
                    print(crop_output.shape, h,w, 160-h, 160-w)
                assert crop_output.shape == (side*2, side*2) or crop_label.shape == (side * 2, side*2) or crop_image.shape == (side * 2, side*2)
                np.save(osp.join(to_input_dir, fn[:-4] + '_0_'+str(idx)+'.npy'), crop_image)
                np.save(osp.join(to_label_dir, fn[:-4] + '_0_'+str(idx)+'.npy'), crop_label)
                np.save(osp.join(to_output_dir, fn[:-4] + '_0_'+str(idx)+'.npy'), crop_output)


        # right
        output1 = output[1]
        rp1 = regionprops(output1)
        if len(rp1) == 0:
            pass
        else:
            for idx, r in enumerate(rp1):
                cx, cy = r.centroid
                cx, cy = int(cx), int(cy)
                x1, x2, y1, y2 = max(0, cx - side), min(512, cx + side), max(0, cy - side), min(512, cy + side)
                h, w = x2 - x1, y2 - y1
                # if h != 160 or w != 160:
                #     print(h,w)
                crop_image = image[x1:x2, y1:y2]
                crop_label = label[x1:x2, y1:y2]
                crop_output = output1[x1:x2, y1:y2]

                aa = np.zeros((160, 160))
                bb = np.zeros((160, 160))
                cc = np.zeros((160, 160))
                aa[:crop_image.shape[0], :crop_image.shape[1]] = crop_image
                bb[:crop_image.shape[0], :crop_image.shape[1]] = crop_label
                cc[:crop_image.shape[0], :crop_image.shape[1]] = crop_output
                crop_image = aa.astype(np.int16)
                crop_label = bb.astype(np.uint8)
                crop_output = cc.astype(np.uint8)

                if crop_output.shape != (side*2, side*2) or crop_label.shape != (side * 2, side*2) or crop_image.shape != (side * 2, side*2):
                    print(crop_output.shape, h, w, 160 - h, 160 - w)
                assert crop_output.shape == (side*2, side*2) or crop_label.shape == (side * 2, side*2) or crop_image.shape == (side * 2, side*2)
                np.save(osp.join(to_input_dir, fn[:-4] + '_1_' + str(idx) + '.npy'), crop_image)
                np.save(osp.join(to_label_dir, fn[:-4] + '_1_' + str(idx) + '.npy'), crop_label)
                np.save(osp.join(to_output_dir, fn[:-4] + '_1_' + str(idx) + '.npy'), crop_output)