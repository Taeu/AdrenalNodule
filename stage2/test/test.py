from utils import make_parser
import os
from dataloader import data_loader
from networks import UNet_
import segmentation_models_pytorch as smp
from losses import DiceLoss
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score
from tqdm import tqdm
import torch
import pickle5 as pickle
import pandas as pd
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def test(model, test_dataloader, criterion,  args):
    model.eval()
    dice_scores = []
    fn_maxProb_dict = {}
    fn_label_dict = {}
    with torch.no_grad():
        for iter_, test_data in tqdm(enumerate(test_dataloader)):
            image, label, label2, fn = test_data
            fn = fn[0]
            label = label.float()
            label2 = label2.float()
            # fetch train data
            if args.cuda:
                image = image.cuda()
                label = label.cuda()
            if args.class_head :
                pred_mask, pred_class = model(image)
                pred_class = pred_class.cpu().numpy()
            else:
                pred_mask = model(image)
            loss = criterion(pred_mask, label)
            dice_score = 1 - loss
            pred_mask = pred_mask.cpu().numpy()
            dice_score = dice_score.cpu().numpy()
            label2 = label2.cpu().numpy()
            if args.class_head == False:
                pred_class = np.max(pred_mask)
            else:
                pred_class = pred_class[0]
            isLeft = '_0_' in fn
            if isLeft:
                fn = fn.split('_')[0] + '_' + fn.split('_')[1] + '_0_'
            else:
                fn = fn.split('_')[0] + '_' + fn.split('_')[1] + '_1_'
            if fn not in fn_maxProb_dict:
                fn_maxProb_dict[fn] = 0
                fn_label_dict[fn] = 0
            fn_maxProb_dict[fn] = max(sigmoid(pred_class), fn_maxProb_dict[fn])
            fn_label_dict[fn] = max(label2, fn_label_dict[fn])
            dice_scores.append(dice_score)

    return fn_maxProb_dict, fn_label_dict, np.mean(dice_scores)

def get_roc(fn_maxProb_dict, gt_dict, side = 'left'):

    if side == 'left':
        new_dict = {k.split('_')[0] + '_' + k.split('_')[1] + '.npy':v for k,v in fn_maxProb_dict.items() if '_1_' in k}
    else:
        new_dict = {k.split('_')[0] + '_' + k.split('_')[1] + '.npy':v for k, v in fn_maxProb_dict.items() if '_0_' in k}
    preds = []
    gts = []
    for fn in gt_dict:
        if fn not in new_dict:
            preds.append(0)
        else:
            preds.append(new_dict[fn])
        gts.append(gt_dict[fn] > 10)

    roc = roc_auc_score(gts, preds)

    print('{} label len : {}'.format(side, np.sum(gts)))
    print('total len : {}'.format(len(gts)))
    print('{} roc {} : {}'.format(side, 'pb', roc))
    ######################
    fpr, tpr, thresholds = roc_curve(gts, preds)
    J = tpr - fpr
    idx = np.argmax(J)
    best_threshold = thresholds[idx]
    print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_threshold, tpr[idx], 1 - fpr[idx], J[idx]))

    new_preds = []
    for pred in preds:
        if pred >= best_threshold:
            new_preds.append(True)
        else:
            new_preds.append(False)
    new_preds = np.array(new_preds)
    gts = np.array(gts)
    print(f'accuracy : {np.sum(new_preds == gts) / len(new_preds)}')
    print(f'FPs : {np.sum((new_preds == True) & (gts == False))}')
    print(f'FNs : {np.sum((new_preds == False) & (gts == True))}')



if __name__ == '__main__':
    args = make_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    test_dataloader = data_loader(args=args, phase='test', batch_size=args.eval_batch_size)

    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    ckpt_path = './ag/result/stage2_0916/checkpoints/det2_set1_dc_and_fc_with_input_160/best_epoch.pth'
    if args.class_head :
        model = UNet_(model, 1)
        ckpt_path = './ag/result/stage2_0916/checkpoints/1224_det2_set1_dc_and_fc_with_input_160/best_epoch.pth'

    model.load_state_dict(torch.load(ckpt_path)['model'])
    model = model.cuda()
    criterion = DiceLoss() # 1 - dice

    fn_maxProb_dict, fn_label_dict, dice_score = test(model, test_dataloader, criterion, args)
    print('dice_score :',dice_score)
    with open(os.path.join('./ag/data/', 'fn_gt_3_0826.pkl'), 'rb') as f:
        gt0 = pickle.load(f) # gt0 = pd.read_pickle('./ag/data/' + 'fn_gt_3_0826.pkl')

    with open(os.path.join('./ag/data/', 'fn_gt_4_0826.pkl'), 'rb') as f:
        gt1 = pickle.load(f) # gt1 = pd.read_pickle('./ag/data/' + 'fn_gt_4_0826.pkl')

    with open('./ag/code_0906/mulan_det_c4_0906/0905_set1.pkl', 'rb') as f:
        check_df = pickle.load(f)
        
    #check_df = pd.read_pickle('./ag/code_0906/mulan_det_c4_0906/0905_set1.pkl')
    tvt_dict = {row.File_name: row.Train_val_test for index, row in check_df.iterrows()}
    gt0 = {k:v for k,v in gt0.items() if tvt_dict[k] == 3}
    gt1 = {k: v for k, v in gt1.items() if tvt_dict[k] == 3}
    print(len(gt0), len(gt1))
    get_roc(fn_maxProb_dict, gt0, 'right')
    get_roc(fn_maxProb_dict, gt1, 'left')

# python test.py --eval_batch_size 1 --class_head gpu_id 0
# python test.py --eval_batch_size 1 --class_head --gpu_id 7 --image_dir stage1_input_det2_set1_epoch_82_train --label_dir stage1_label_det2_set1_epoch_82_train --output_dir stage1_output_det2_set1_epoch_82_train
# python test.py --eval_batch_size 1 --gpu_id 7 --image_dir stage1_input_det2_set1_epoch_82_train --label_dir stage1_label_det2_set1_epoch_82_train --output_dir stage1_output_det2_set1_epoch_82_train