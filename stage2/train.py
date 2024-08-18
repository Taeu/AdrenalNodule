import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
# from torch.utils.tensorboard import SummaryWriter
import json
import shutil
import torch.nn.functional as F
from dataloader import data_loader, data_loader3
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import random

# model
import segmentation_models_pytorch as smp
import torchvision.models as models
# from networks import UNet_

# losses
from losses import DiceLoss, BinaryFocalLoss
from losses2 import DC_and_Focal_loss

def validate(model, validate_dataloader, criterion, criterion2, args):
    model.eval()
    epoch_loss = []
    with torch.no_grad():
        for iter_, test_data in tqdm(enumerate(validate_dataloader)):
            image, label, label2 = test_data
            label = label.float()
            label2 = label2.float()
            # fetch train data
            if args.cuda:
                image = image.cuda()
                label = label.cuda()
            #pred_label, pred_class = model(image)
            pred = model(image)
            # if 'hrnet' in args.model_name:
            #     loss = criterion(up4(pred[1]), label)
            #     #loss += criterion(up4(pred[1]), label)
            # else:
            loss = criterion(pred, label)
            #loss += criterion2(pred_class, label2)
            epoch_loss.append(float(loss))

    return np.mean(epoch_loss)

def save_model(checkpoint_dir, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        #'optimizer':optimizer.state_dict(),
        #'scheduler':scheduler.state_dict(),
    }
    torch.save(state, os.path.join(checkpoint_dir + '.pth'))
    print('model saved')

def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')

def make_parser():
    args = argparse.ArgumentParser()
    #exp
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--save_dir", type=str, default='./ag/result/stage2/') # ./ag/result/stage2_0916/
    args.add_argument("--exp_name", type=str, default = 'unet')
    args.add_argument("--train_batch_size", type=int, default=64)
    args.add_argument("--eval_batch_size", type=int, default=128)
    args.add_argument("--cuda", type=bool, default=True)

    #dataset
    args.add_argument("--root", type=str, default='./ag/data/stage2/') # ./ag/data/stage2_0916/
    args.add_argument("--image_dir", type=str, default='input') # stage1_input_train_unet_crop
    args.add_argument("--label_dir", type=str, default='label') # stage1_label_train_unet_crop
    args.add_argument("--output_dir", type=str, default='output') # stage1_output_train_unet_crop

    args.add_argument("--image_size", type=int, default=160)
    args.add_argument("--advprop", type=bool, default=False)

    #model
    args.add_argument("--model_name", type=str, default="unet")
    args.add_argument('--squeeze', action='store_true', default=False)
    args.add_argument("--transfer", type=str, default=None)
    args.add_argument("--dropout", type=float, default=0.2)
    args.add_argument("--num_classes", type=int, default=1)
    #hparams
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--step", type=float, default=15)
    args.add_argument("--num_epochs", type=int, default=60)
    # args.add_argument("--val_same_epoch", type=int, default=20)
    args.add_argument("--weight_decay", type=float, default=1e-5)
    args.add_argument("--optim", type=str, default="rangerlars")
    args.add_argument("--scheduler", type=str, default="cosine")
    args.add_argument("--warmup", type=int, default=5)
    args.add_argument("--cutmix_alpha", type=float, default=1)
    args.add_argument("--cutmix_prob", type=float, default=0.5)

    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--gpu_id", type=str, default="0")
    args.add_argument('--focal_loss', action='store_true', default=False)
    args.add_argument('--no_weights', action='store_true', default=False)
    args.add_argument('--amp', action='store_true', default=False)
    args.add_argument('--data_aug', action='store_true', default=False)

    args.add_argument("--test_for_train", type=bool, default=False)
    config = args.parse_args()
    return config

# mulan det2
# python train.py --image_dir stage1_input_train_det2_crop --label_dir stage1_label_train_det2_crop  --output_dir stage1_output_train_det2_crop --exp_name det2_dc_and_fc_with_input_160 --model_name unet --gpu_id 3
# python train.py --image_dir stage1_input_train_det2_crop --label_dir stage1_label_train_det2_crop  --output_dir stage1_output_train_det2_crop --exp_name det2_dc_and_fc_with_input_160_2 --model_name unet --gpu_id 4
# 231112/02:52
# python train.py --exp_name det2_dc_and_fc_with_input_160 --model_name unet --gpu_id 0
# python train.py --exp_name det2_dc_and_fc_with_input_160_aug --model_name unet --gpu_id 1 --data_aug

# python train.py --exp_name det2_dc_and_fc_with_input_160_finetune10 --model_name unet --gpu_id 3 --num_epochs 10 --lr 1e-5 --transfer ./ag/result/stage2_0916/checkpoints/det2_set1_dc_and_fc_with_input_160/best_epoch.pth 


# from models.seg_hrnet_ocr import *
from config import config
from config import update_config
import yaml
up4 = nn.Upsample(scale_factor=4, mode='nearest')

def main(args):
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    #set_seed(args.seed)
    # args = vars(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # For gpu
    DATASET_PATH = args.root
    torch.manual_seed(args.seed)
    #make dataset
    if args.data_aug:
        train_dataloader = data_loader3(args=args, phase='train', batch_size=args.train_batch_size)
        validate_dataloader = data_loader3(args=args, phase='valid', batch_size=args.eval_batch_size)
    else:
        train_dataloader = data_loader(args=args, phase='train', batch_size=args.train_batch_size)
        validate_dataloader = data_loader(args=args, phase='valid', batch_size=args.eval_batch_size)
    # test_dataloader = data_loader(args=args, phase='test', batch_size=args.eval_batch_size)
    
    #make model
    if 'unet' in args.model_name:
        encoder_name = 'resnet34'
        if args.squeeze:
            encoder_name = 'se_resnet50'
        model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
    else:
        if '18' in args.model_name:
            model = models.resnet18(pretrained=True)
        elif '34' in args.model_name:
            model = models.resnet34(pretrained=True)
        if 'res' in args.model_name:
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=14, bias=True)

    model = model.cuda() if args.cuda else model

    if args.transfer is not None:
        model.load_state_dict(torch.load(args.transfer)['model'])


    # add sigmoide
    #criterion = torch.nn.CrossEntropyLoss()
    if 'focal' in args.exp_name:
        criterion = BinaryFocalLoss()
    elif 'dice' in args.exp_name :
        criterion = DiceLoss()
    elif 'dc_and_fc' in args.exp_name:
        criterion = DC_and_Focal_loss({'batch_dice':True, 'smooth':1e-5,'do_bg':False}, {'alpha':0.5, 'gamma':2, 'smooth':1e-5})
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    criterion2 = torch.nn.BCEWithLogitsLoss()
    criterion = criterion.cuda() if args.cuda else criterion
    criterion2 = criterion2.cuda() if args.cuda else criterion2
    optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=args.step, gamma=0.5)

    log_dir = os.path.join(args.save_dir, 'logs', args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints',args.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # set information
    start_time = datetime.datetime.now()
    num_batches = len(train_dataloader)
    
    #check parameter of model
    print("------------------------------------------------------------")
    total_params = sum(p.numel() for p in model.parameters())
    print("num of parameter : ",total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of trainable_ parameter :",trainable_params)
    print("num batches :",num_batches)
    print("------------------------------------------------------------")

    # train
    global_iter = 0
    val_global_iter = 0
    max_val_score = 0
    for epoch in tqdm(range(args.num_epochs)):
        # -------------- train -------------- #
        model.train()
        epoch_loss = []
        
        for iter_, train_data in enumerate(train_dataloader):
            image, label, label2 = train_data
            label = label.float()
            label2 = label2.float()
            # fetch train data
            if args.cuda:
                image = image.cuda()
                label = label.cuda()
                label2 = label2.cuda()

            #pred_label, pred_class = model(image)
            pred = model(image)
            loss = criterion(pred, label)
            #loss += criterion2(pred_class, label2.unsqueeze(-1))
            # loss backward
            loss.backward()
            
            epoch_loss.append(float(loss))
            if iter_ % 50 == 0:
                print('Epoch: {} | Iteration: {} | Running loss: {:1.5f}'.format(epoch, iter_, np.mean(np.array(epoch_loss))))
            
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()

            global_iter+=1
            #break
        # scheduler update
        scheduler.step()
       
            
        eval_score = validate(model, validate_dataloader, criterion, criterion2, args)
        print(epoch,' epcoh eval_score : ', 1-eval_score)
        if max_val_score < 1-eval_score:
            max_val_score = 1-eval_score
            model_save_dir = os.path.join(checkpoint_dir, "best_epoch")
            save_model(model_save_dir, model, optimizer, scheduler)

    print('==================== end ============================')
    print(args.exp_name, 'Max val score : ',max_val_score)
    print('==================== end ============================')

if __name__ == '__main__':
    # mode argument
    tic = time.time()
    args = make_parser()
    main(args)
    toc = time.time()
    print('elapsed time : {} Sec'.format(round(toc - tic, 3)))