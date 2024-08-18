from run import *
from maskrcnn.engine.processor import *
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import os
import numpy as np
import pdb
def get_args():
    parser = argparse.ArgumentParser(description='Mulan Eval')

    parser.add_argument('--num_gpu', type=str, default='1')
    parser.add_argument('--from_pkl', type=str, default='AG_AGN_1201_2') # AG_AGN2'
    parser.add_argument('--to_pkl', type=str, default='AG_AGN_1201_2')
    parser.add_argument('--version', type=str, default='mulan_det_agagn_1201_2_116') # 'mulan_det_c2_1201_2_57', # 'mulan_det_c4_1201_2_110', 'mulan_det_c4_1201_2_110
    parser.add_argument('--train', type=str, default='f')  # version 고칠때 그에 맞게 deeplesion.py 에서 class 도 가져와야함.
    parser.add_argument('--made_val_pickle_1201', type=str, default='f') # 'f'
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.runtime_info.distributed = num_gpus > 1

    return args

def load_model(args):
    config_file = 'config.yml'
    cfg_new = cfg_from_file(config_file)
    merge_a_into_b(cfg_new, cfg)
    if cfg.MODE in ('demo', 'batch'):
        cfg_test = merge_test_config()
    check_configs()
    cfg.runtime_info.local_rank = 0
    cfg.MODE = 'val'
    datasets = make_datasets('val')
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
    name = 'checkpoints/' + args.version + '.pth'
    extra_checkpoint_data = checkpointer.load(name)


if __name__ == '__main__':

    args = get_args()
    # train용 eval
    # test용 eval


    model = load_model(args)
    test_model(model, True)

    # input : test 에 필요한 pickle 을 넣고 from_pkl, to_pkl 로 각각 만들고, to_pkl 로 돌아가게 만들기
    # 근데 이거는 미리 만들어놓으면 되니까.