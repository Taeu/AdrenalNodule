# save stage1 output with label
import pandas as pd 
import argparse
from tqdm import tqdm
import numpy as np
from skimage.measure import regionprops
import os

def get_args():
    
    args = argparse.ArgumentParser()
    args.add_argument("--pickle_path", type=str, default='./ag/result/stage1/snu_snub.pkl')
    args.add_argument("--input_dir", type=str, default='./ag/data/data1/Numpy2/')
    args.add_argument("--label_dir", type=str, default='./ag/data/data1/Label_npy_int/')
    args.add_argument("--save_root_dir", type=str, default='./ag/data/stage2/')
    
    args.add_argument("--start_idx", type=int, default=0)
    args.add_argument("--end_idx", type=int, default=16000)

    args = args.parse_args()
    return args


def save_data(output, image, label, fname, sidx, left_right, side=80):
    rp = regionprops(output)
    if len(rp)==0:
        return
    
    r = rp[0]
    cx, cy = r.centroid
    cx,cy = int(cx), int(cy)
    x1, x2, y1, y2 = max(0, cx-side), min(512, cx+side), max(0,cy-side), min(512, cy+side)
    h, w = x2-x1, y2-y1
    crop_image = image[x1:x2,y1:y2]
    crop_label = label[x1:x2,y1:y2] >= 3
    crop_output = output[x1:x2,y1:y2]
    
    aa = np.zeros((160,160))
    bb = np.zeros((160,160))
    cc=  np.zeros((160,160))
    aa[:crop_image.shape[0], :crop_image.shape[1]] = crop_image
    bb[:crop_image.shape[0], :crop_image.shape[1]] = crop_label
    cc[:crop_image.shape[0], :crop_image.shape[1]] = crop_output
    crop_image = aa.astype(np.int16)
    crop_label = bb.astype(np.uint8)
    crop_output = cc.astype(np.uint8)
    
    if crop_output.shape != (side*2, side*2) or crop_label.shape != (side * 2, side*2) or crop_image.shape != (side * 2, side*2):
        print(crop_output.shape, h,w, 160-h, 160-w)
    assert crop_output.shape == (side*2, side*2) or crop_label.shape == (side * 2, side*2) or crop_image.shape == (side * 2, side*2)
    np.save(os.path.join(args.save_root_dir,'input', fname[:-4] + left_right + str(sidx)+'.npy'), crop_image)
    np.save(os.path.join(args.save_root_dir,'label', fname[:-4] + left_right + str(sidx)+'.npy'), crop_label)
    np.save(os.path.join(args.save_root_dir,'output', fname[:-4] + left_right + str(sidx)+'.npy'), crop_output)



if __name__=='__main__':
    
    args = get_args()
    
    os.makedirs(os.path.join(args.save_root_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(args.save_root_dir, 'label'), exist_ok=True)
    os.makedirs(os.path.join(args.save_root_dir, 'output'), exist_ok=True)
    
    df = pd.read_pickle(args.pickle_path)
    print(df.shape)
    
    
    for index, row in tqdm(df[args.start_idx:args.end_idx].iterrows()):
        
        try:
            if len(row.bbox) == 0:
                continue
            cur_slice = np.zeros((2,512,512))
            for b in row.bbox:
                y_middle = (b[1] + b[3])/2
                idx = 0
                if y_middle > 256:
                    idx = 1
                # if (b[3] - b[1]) > 120 and b[3] > 256 and b[1] < 256 and (b[2] - b[0]) < 60: ## added for post-processing cases such as aorta
                #     continue
                cur_slice[idx, int(b[0]):int(b[2]), int(b[1]): int(b[3])] = 1
            cur_slice = cur_slice.astype(np.uint8)
            
            image = np.load(args.input_dir + row.fns)[:,:,row.idx]
            label = np.load(args.label_dir + row.fns)[:,:,row.idx]
            
            save_data(cur_slice[0], image, label, row.fns, row.idx, left_right = '_0_')
            save_data(cur_slice[1], image, label, row.fns, row.idx, left_right = '_1_')
        except Exception as e :
            print(row, e)
        
        