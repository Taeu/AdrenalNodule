from run import *
from maskrcnn.engine.processor import *
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import os
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Mulan Eval')
    parser.add_argument('--num_gpu', type=str, default='0')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--version', type=str,default='det2_set1_epoch_82')  # 'det_agagn_1201_2_116' 'mulan_seg_agagn_1201_1_epoch_54', 'mulan_det_c2_1201_2_57', # 'mulan_det_c4_1201_2_110', 'mulan_det_c4_1201_2_110
    parser.add_argument('--data_dir', type=str, default='./ag/data5/numpy_new/') ##
    parser.add_argument('--result_dir', type=str, default='./ag/result/1/') ##
    args = parser.parse_args()
    return args


# python eval.py --pkl AG_AGN_1201_1 --version mulan_det_c2_66_train --train t --num_gpu 6
# python eval_gc.py --version mulan_det_c2_66
# python eval_gc.py --version det2_set1_epoch_82 --num_gpu 0 --data_dir ./ag/data/Numpy2/ --start_idx 0 --end_idx 250

# python eval_gc.py --version det2_set1_epoch_82 --num_gpu 0 --data_dir ./ag/data3/numpy/ --start_idx 0 --end_idx -1
# python eval_gc.py --version det4_set1_epoch_85 --num_gpu 0 --data_dir ./ag/data/Numpy2/ --start_idx 0


def windowing(im, win=[-350, 350]):
    """scale intensity from win[0]~win[1] to float numbers in 0~255"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

def get_input(input_npy: [100, 512, 512], idx: 0):
    input = np.zeros((3, 3, 512, 512))
    max_idx = input_npy.shape[0]

    cnt = 0
    for i in range(-4, -1):
        new_idx = idx + i
        if new_idx < 0:
            new_idx = 0
        elif new_idx >= max_idx:
            new_idx = max_idx - 1
        input[0, cnt] = input_npy[new_idx]
        cnt += 1

    cnt = 0
    for i in range(-1, 2):
        new_idx = idx + i
        if new_idx < 0:
            new_idx = 0
        elif new_idx >= max_idx:
            new_idx = max_idx - 1
        input[1, cnt] = input_npy[new_idx]
        cnt += 1

    cnt = 0
    for i in range(2, 5):
        new_idx = idx + i
        if new_idx < 0:
            new_idx = 0
        elif new_idx >= max_idx:
            new_idx = max_idx - 1
        input[2, cnt] = input_npy[new_idx]
        cnt += 1

    input = windowing(input)
    input /= 255.
    input = torch.from_numpy(input).to(dtype=torch.float)
    # input = input.unsqueeze(0)

    return input

def post_bbox(bbox: "BoxList"):
    new_b = []
    new_s = []
    new_l = []
    b = bbox.bbox.numpy()
    s = bbox.get_field('scores').numpy()
    l = bbox.get_field('labels').numpy()
    for x, y, z in zip(b, s, l):
        if y >= 0.5:
            new_b.append(x)
            new_s.append(y)
            new_l.append(z)

    return new_b, new_s, new_l

def get_model():
    args = get_args()
    config_file = 'config.yml'
    cfg_new = cfg_from_file(config_file)
    merge_a_into_b(cfg_new, cfg)
    log_dir = cfg.LOGFILE_DIR
    # logger = setup_logger("maskrcnn", log_dir, cfg.EXP_NAME, get_rank())

    if cfg.MODE in ('demo', 'batch'):
        cfg_test = merge_test_config()
        # logger.info(pprint.pformat(cfg_test))
    else:
        pass
    # logger.info("Loaded configuration file {}".format(config_file))
    # logger.info(pprint.pformat(cfg_new))
    check_configs()
    cfg.runtime_info.local_rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu  # cfg.GPU
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.runtime_info.distributed = num_gpus > 1
    # logger.info("Using {} GPUs".format(num_gpus))
    cfg.MODE = 'val'
    model = build_detection_model()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    optimizer = make_optimizer(model)
    arguments = {}
    arguments['start_epoch'] = cfg.BEGIN_EPOCH
    arguments['max_epoch'] = cfg.SOLVER.MAX_EPOCH
    cfg.EXP_NAME = args.version
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(model, optimizer, scheduler=None, save_dir=cfg.CHECKPT_DIR,
                                         prefix=cfg.EXP_NAME, save_to_disk=save_to_disk)
    if 'det4' in args.version:
        name = './ag/code_0906/mulan_det_c4_0906/checkpoints/' + args.version + '.pth'
    elif 'det2' in args.version:
        name = './code_0906/mulan_det_c2_0906/checkpoints/' + args.version + '.pth'
    extra_checkpoint_data = checkpointer.load(name)

    device = 'cuda'
    device = torch.device(device)
    model.eval()
    model.to(device)
    return model, args, device

if __name__ == '__main__':

    model, args, device = get_model()
    cpu_device = torch.device('cpu')

    data_dir = args.data_dir #
    name = 'GC'
    check_dict = None
    if 'Numpy2' in data_dir:
        name = 'SNUB'
        check_df = pd.read_pickle('./ag/code_0906/mulan_det_c4_0906/0905_set1.pkl')
        check_dict = {row.File_name:row.Train_val_test for index, row in check_df.iterrows()}
    # new GC add !

    os.makedirs(args.result_dir, exist_ok=True)
    if args.result_dir[-1] != '/':
        args.result_dir += '/'
    ll = sorted(os.listdir(data_dir))[args.start_idx:args.end_idx]
    flist = []; tvt_list = []; idx_list = []; bbox_list = []; score_list = []; label_list = [];


    with torch.no_grad():
        for fns in tqdm(ll):
            tvt = 3
            if check_dict is not None:

                if fns not in check_dict.keys():
                    continue
                tvt = check_dict[fns]
            a = np.load(data_dir + fns)
            if 'Numpy2' in data_dir or 'numpy' in data_dir:
                a = np.transpose(a, (2,0,1))
            total_outputs = []
            for j in range(a.shape[0]):
                cur_input = get_input(a, j)
                cur_input = cur_input.to(device)
                outputs = model(cur_input, None, None)
                outputs = [o.to(cpu_device) for o in outputs]
                total_outputs.append(outputs)

            for idx, bbox in enumerate(total_outputs):
                if len(bbox[0]) != 0:
                    bbox, scores, labels = post_bbox(bbox[0])
                    if len(bbox) != 0:
                        flist.append(fns)
                        idx_list.append(idx)
                        bbox_list.append(bbox)
                        score_list.append(scores)
                        label_list.append(labels)
                        tvt_list.append(tvt)

    df = pd.DataFrame({'fns': flist, 'tvt':tvt_list,'idx': idx_list, 'bbox': bbox_list, 'score': score_list, 'label': label_list,})
    df.to_pickle('{}{}_{}.pkl'.format(args.result_dir, args.version, args.start_idx))
    # df.to_csv('./results/GC_STAGE1.csv')
    # test_model(model, True)

