from run import *
from maskrcnn.engine.processor import *
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import os
import numpy as np
def get_args():
    parser = argparse.ArgumentParser(description='Mulan Eval')

    parser.add_argument('--num_gpu', type=str, default='1')
    parser.add_argument('--pkl', type=str, default='AGN_val_stage2_0527.pkl') #'AGN_val_0527.pkl' 'AG_AGN_1201_2' # AG_AGN2'
    parser.add_argument('--version', type=str, default='det_c4_55_test') # 'det_agagn_1201_2_116' 'mulan_seg_agagn_1201_1_epoch_54', 'mulan_det_c2_1201_2_57', # 'mulan_det_c4_1201_2_110', 'mulan_det_c4_1201_2_110
    parser.add_argument('--train', type=str, default='f')  # version 고칠때 그에 맞게 deeplesion.py 에서 class 도 가져와야함.
    parser.add_argument('--made_val_pickle_0527', type=str, default='f') # 'f'
    args = parser.parse_args()

    return args
# python eval.py --pkl AG_AGN_1201_1 --version mulan_det_c2_66_train --train t --num_gpu 6
def to_val_pickle(pkl_name, is_train = False):
    print('start to make AG_AGN_val!')
    origin_pd = pd.read_pickle(pkl_name)
    opd_tvt = origin_pd['Train_val_test'].tolist()
    for i in range(len(opd_tvt)):
        if opd_tvt[i] == 3:
            opd_tvt[i] = 2
    origin_pd['Train_val_test'] = opd_tvt
    val_id = 2
    if is_train:
        val_id = 1
    val_pd = origin_pd[origin_pd['Train_val_test'] == val_id]

    if is_train :
        opd_tvt = val_pd['Train_val_test'].tolist()
        for i in range(len(opd_tvt)):
            if opd_tvt[i] == 1:
                opd_tvt[i] = 2
        val_pd['Train_val_test'] = opd_tvt

    val_pd_filename = list(set(val_pd['File_name'].tolist()))
    # 0527 edit
    with open('fn_len.pkl', 'rb') as handle:
        fn_maxlen_dict = pickle.load(handle)
    
    fnl = []; ksi = []; sl = []; lil = []; tvtl = []; spl = []; sintv = []; bboxl = []
    for fn in val_pd_filename:
        cur_df = val_pd[val_pd['File_name'] == fn]
        slice_idx_exist = set(cur_df['Key_slice_index'].tolist())
        slice_idx_new = set(np.arange(0, 100))#set(np.arange(0, fn_maxlen_dict[fn]))
        slice_idx_new = list(slice_idx_new - slice_idx_exist)
        for i in slice_idx_new :
            fnl.append(fn)
            ksi.append(i)
            sl.append([0])
            lil.append([0])
            tvtl.append(2)
            spl.append([0.67, 0.67])
            sintv.append(2.0)
            bboxl.append([[0,0,0,0]])

    d = {'File_name': fnl , 'Key_slice_index': ksi , 'size': sl , 'label_id':lil , 'Train_val_test':tvtl, 'Spacing' : spl, 'Slice_intv':sintv, 'bbox_list':bboxl}
    df_new = pd.DataFrame(data=d)
    frames = [val_pd, df_new]
    result = pd.concat(frames)
    result = result.sort_values(by=['File_name', 'Key_slice_index'])
    result = result.reset_index(drop=True)
    if is_train :
        result.to_pickle('AGN_val_0527_train.pkl')
    else:
        result.to_pickle('AGN_val_0527_all.pkl')
    print('made AG_AGN_val!')

def to_val_pickle_1201(pkl_name, is_train=False):
    print('start to make AG_AGN_1201_2_val!')
    origin_pd = pd.read_pickle(pkl_name)
    opd_tvt = origin_pd['Train_val_test'].tolist()
    for i in range(len(opd_tvt)):
        if opd_tvt[i] == 2:
            opd_tvt[i] = 1
        elif opd_tvt[i] == 3:
            opd_tvt[i] = 2

    origin_pd['Train_val_test'] = opd_tvt
    val_id = 2
    if is_train:
        val_id = 1
    val_pd = origin_pd[origin_pd['Train_val_test'] == val_id]

    if is_train:
        opd_tvt = val_pd['Train_val_test'].tolist()
        for i in range(len(opd_tvt)):
            if opd_tvt[i] == 1:
                opd_tvt[i] = 2
        val_pd['Train_val_test'] = opd_tvt

    all_pd_filename = list(set(origin_pd['File_name'].tolist()))
    val_pd_filename = list(set(val_pd['File_name'].tolist()))
    print(len(all_pd_filename), len(val_pd_filename))
    label_npy_path = './ag/data/Label_npy/'
    fn_maxlen_dict = {}
    cnt = 0
    # for fn in all_pd_filename:
    #     cnt += 1
    #     if cnt % 50 == 0:
    #         print(cnt, end=',')
    #     label_npy = np.load(os.path.join(label_npy_path, fn))
    #     fn_maxlen_dict[fn] = label_npy.shape[-1]

    # with open('fn_maxlen_dict_1201_2.pkl', 'wb') as handle:
    #     pickle.dump(fn_maxlen_dict, handle) #, protocol=pickle.HIGHEST_PROTOCOL)

    with open('fn_maxlen_dict.pkl', 'rb') as handle:
        fn_maxlen_dict = pickle.load(handle)

    fnl = [];
    ksi = [];
    sl = [];
    lil = [];
    tvtl = [];
    spl = [];
    sintv = [];
    bboxl = []
    for fn in val_pd_filename:
        cur_df = val_pd[val_pd['File_name'] == fn]
        slice_idx_exist = set(cur_df['Key_slice_index'].tolist())
        slice_idx_new = set(np.arange(0, 100))#set(np.arange(0, fn_maxlen_dict[fn]))
        slice_idx_new = list(slice_idx_new - slice_idx_exist)
        for i in slice_idx_new:
            fnl.append(fn)
            ksi.append(i)
            sl.append([0])
            lil.append([0])
            tvtl.append(2)
            spl.append([0.67, 0.67])
            sintv.append(2.0)
            bboxl.append([[0, 0, 0, 0]])

    d = {'File_name': fnl, 'Key_slice_index': ksi, 'size': sl, 'label_id': lil, 'Train_val_test': tvtl, 'Spacing': spl,
         'Slice_intv': sintv, 'bbox_list': bboxl}
    df_new = pd.DataFrame(data=d)
    frames = [val_pd, df_new]
    result = pd.concat(frames)
    result = result.sort_values(by=['File_name', 'Key_slice_index'])
    result = result.reset_index(drop=True)
    if is_train:
        result.to_pickle('AG_AGN_val_train_1201_1')
    else:
        result.to_pickle('AG_AGN_val_1201_1')
    # val_pd.to_pickle('AG_AGN_val')
    print('made AG_AGN_val_1201_2!')
    # df.to_pickle('AG_AGN_val')

if __name__ == '__main__':

    args = get_args()
    if args.train == 't':
        to_val_pickle(args.pkl, is_train = True)
    if args.made_val_pickle_0527 == 't':
        to_val_pickle(args.pkl) # args.pkl : AG_AGN_1201_2_val
    config_file = 'config.yml'
    cfg_new = cfg_from_file(config_file)
    merge_a_into_b(cfg_new, cfg)
    log_dir = cfg.LOGFILE_DIR
    logger = setup_logger("maskrcnn", log_dir, cfg.EXP_NAME, get_rank())

    if cfg.MODE in ('demo', 'batch'):
        cfg_test = merge_test_config()
        logger.info(pprint.pformat(cfg_test))
    else:
        logger.info("Loaded configuration file {}".format(config_file))
        logger.info(pprint.pformat(cfg_new))
    check_configs()
    cfg.runtime_info.local_rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.num_gpu#cfg.GPU
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.runtime_info.distributed = num_gpus > 1
    logger.info("Using {} GPUs".format(num_gpus))
    cfg.MODE = 'val'
    if cfg.MODE in ('train',) and cfg.SEED is not None:
        random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        cudnn.deterministic = True
        np.random.seed(cfg.SEED)
        logger.info('Manual random seed %s', cfg.SEED)
    
    logger = logging.getLogger('maskrcnn.test')
    datasets = make_datasets('val')
    logger.info('building model ...')
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
    test_model(model, True)

