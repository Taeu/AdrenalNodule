import sys
sys.path.append('./')
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import pickle5 as pickle
# model
import segmentation_models_pytorch as smp
# losses
from losses import DiceLoss
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
    args.add_argument("--pkl_name", type=str, default='snu_snub.pkl') # det2_set1_epoch_82_0.pkl
    args.add_argument("--exp_name", type=str, default = 'det2_dc_and_fc_with_input_160') # det2_set1_dc_and_fc_with_input_160
    args.add_argument("--model_name", type=str, default='unet')
    args.add_argument("--cuda", type=bool, default=True)

    #dataset
    args.add_argument("--data_dir", type=str, default='./ag/data/data1/Numpy2/')
    args.add_argument("--pkl_dir", type=str, default='./ag/result/stage1/')
    args.add_argument("--save_dir", type=str, default='./ag/result/stage2/')
    args.add_argument("--checkpoint_path", type=str, default='/stage2_0916/checkpoints/')
    args.add_argument("--image_size", type=int, default=160)
    
    args.add_argument("--start_idx", type=int, default=0)
    args.add_argument("--end_idx", type=int, default=16000)
    
    args.add_argument("--gpu_id", type=str, default="0")
    args.add_argument("--save_result", action='store_true', default=False)
    args.add_argument("--save_output", action='store_true', default=False)
    args.add_argument('--class_head', action='store_true', default=False)
    args.add_argument('--squeeze', action='store_true', default=False)

    config = args.parse_args()
    return config

def sigmoid(x):
    return 1 / (1 +torch.exp(-x))

def windowing(im, win = [-350, 350]):
    """scale intensity from win[0]~win[1] to float numbers in 0~255"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1


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

from config import config
from config import update_config
import yaml
import torch.nn.functional as F

def get_input(image, output, side = 80):
    input = np.zeros((160, 160, 3))
    rp = regionprops(output)
    if len(rp) != 0 :
        r = rp[0]
        cx, cy = r.centroid
        cx, cy = int(cx), int(cy)
        x1, x2, y1, y2 = max(0, cx - side), min(512, cx + side), max(0, cy - side), min(512, cy + side)
        crop_image = image[x1:x2, y1:y2]
        crop_output = output[x1:x2, y1:y2]
        crop_image_output = crop_image * crop_output
        
        input[:, :, 0] = crop_image
        input[:, :, 1] = crop_image_output
        input[:, :, 2] = crop_image_output
        input = windowing(input)
    input = input.astype(np.uint8)
    input = tfs_val(input).unsqueeze(0)
    return input

def get_output(image, group, model, is_left=True):
    input_array_0 = [] 
    image_array = [] 
    sidx_list = []
    for index, row in group.iterrows():
        cur_slice = np.zeros((512,512)).astype(np.int)
        for b in row.bbox:
            y_middle = (b[1] + b[3])/2
            if (b[3] - b[1]) > 120 and b[3] > 256 and b[1] < 256 and (b[2] - b[0]) < 60: ## added for post-processing cases such as aorta
                continue
            if (is_left == True and y_middle <= 256) or (is_left == False and y_middle >= 256):
                cur_slice[int(b[0]):int(b[2]), int(b[1]): int(b[3])] = 1
        
        input_array_0.append(get_input(image[:,:,int(row.idx)],  cur_slice))
        # input_array_1.append(get_input(image[:,:,int(row.idx)],  cur_slice[1]))
        sidx_list.append(row.idx)
        image_array.append(image[:,:,int(row.idx)])

    input_array_0 = torch.cat(input_array_0).cuda()
    
    with torch.no_grad():
        output_0 = sigmoid(model(input_array_0).detach().cpu()) # (# of slice, 3, 160, 160)
        
    output_0 = F.interpolate(output_0, (40, 40))
    image_array = np.concatenate(image_array)
    return output_0, image_array, sidx_list
# 2개 합치기
def eval_per_fname(args, df_group, model, is_left=True):
    
    
    fname_list = []
    sidx_list_list = []
    max_value_list = []
    max_list = []
    cnt = 0
    for name, group in tqdm(df_group):
        is_left = True
        if is_left:
            save_path = os.path.join(args.save_dir, args.pkl_name.replace('.pkl',''), '0')
        else:
            save_path = os.path.join(args.save_dir, args.pkl_name.replace('.pkl',''), '1')
        
        os.makedirs(save_path, exist_ok=True)
        if args.save_output:
            save_input_path = os.path.join(save_path, 'input')
            os.makedirs(save_input_path, exist_ok=True)    
            save_output_path = os.path.join(save_path, 'output')
            os.makedirs(save_output_path, exist_ok=True)    
        try:
            image = np.load(test_data_dir + name)
            output, image_array, sidx_list  = get_output(image, group, model, is_left= is_left)
            max_values_per_batch = list(torch.max(output.view(output.size(0), -1), dim=1)[0].numpy())
            fname_list.append(name)
            sidx_list_list.append(sidx_list)
            max_value_list.append(max_values_per_batch)
            max_list.append(max(max_values_per_batch))
            if args.save_output:
                # np.save(os.path.join(save_input_path,  name), image_array)
                np.save(os.path.join(save_output_path, name), output.numpy())
            # if cnt > 20: # HACK!
            #     break
            # cnt += 1
        except Exception as e :
            print(name, e)
            
        is_left = False
        if is_left:
            save_path = os.path.join(args.save_dir, args.pkl_name.replace('.pkl',''), '0')
        else:
            save_path = os.path.join(args.save_dir, args.pkl_name.replace('.pkl',''), '1')
        
        os.makedirs(save_path, exist_ok=True)
        if args.save_output:
            save_input_path = os.path.join(save_path, 'input')
            os.makedirs(save_input_path, exist_ok=True)    
            save_output_path = os.path.join(save_path, 'output')
            os.makedirs(save_output_path, exist_ok=True)    
            
        try:
            image = np.load(test_data_dir + name)
            output, image_array, sidx_list  = get_output(image, group, model, is_left= is_left)
            max_values_per_batch = list(torch.max(output.view(output.size(0), -1), dim=1)[0].numpy())
            fname_list.append(name)
            sidx_list_list.append(sidx_list)
            max_value_list.append(max_values_per_batch)
            max_list.append(max(max_values_per_batch))
            if args.save_output:
                # np.save(os.path.join(save_input_path,  name), image_array)
                np.save(os.path.join(save_output_path, name), output.numpy())
            # if cnt > 20: # HACK!
            #     break
            # cnt += 1
        except Exception as e :
            print(name, e)
    
    save_df = pd.DataFrame({'fname':fname_list, 'max_value':max_list, 'sidx_list':sidx_list_list, 'max_value_list':max_value_list})
    save_df['max_value_list'] = save_df['max_value_list'].apply(lambda x: [f'{val:.4f}' for val in x])
    save_df['max_value'] = save_df['max_value'].apply(lambda x: f'{x:.4f}')
    
    save_df.to_excel(os.path.join(save_path, 'result.xlsx'))
    save_df.to_csv(os.path.join(save_path, 'result.csv'), sep='\t')

if __name__ == '__main__':
    args = make_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # For gpu

    # if 'unet' in args.exp_name or args.exp_name == 'det2_set1_dc_and_fc_with_input_160':
    encoder_name = 'resnet34'
    model = smp.Unet(
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    # checkpoint_dir = f'/./ag/result/stage2_0916/checkpoints/{args.exp_name}/best_epoch.pth'
    # checkpoint_dir = f'./ag/result/stage2/checkpoints/{args.exp_name}/best_epoch.pth'
    model = model.cuda()
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])
    model.eval()


    criterion = DiceLoss()
    # args.save_dir = args.pkl_dir
    criterion = criterion.cuda() if args.cuda else criterion
    
    
    test_data_dir = args.data_dir 
    with open(args.pkl_dir + args.pkl_name,'rb') as f:
        df = pickle.load(f)    
    df_group = df.groupby('fns')
    
    
    is_left = True
    eval_per_fname(args, df_group, model, is_left = True)
    # eval_per_fname(args, df_group, model, is_left = False)