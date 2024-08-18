from maskrcnn.data.datasets.evaluation.DeepLesion.DL_eval import *
from maskrcnn.data.datasets.evaluation.DeepLesion.detection_eval import *
from maskrcnn.data.datasets.evaluation.DeepLesion.detection_eval import IOU

import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_auc_score, accuracy_score,auc
from sklearn import metrics
import pandas as pd

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
def nms_taeu_2(boxes,scores, overlapThresh= 0.2):
    total_bbox = []; total_scores= []
    if len(boxes) == 0: return np.array(boxes),np.array(scores)
    total_bbox.append(boxes[0]); boxes = np.delete(boxes, [0], 0)
    total_scores.append(scores[0]); scores = np.delete(scores, [0], 0)
    
    cur_idx = 0
    while(len(boxes) > 0):
        pop_list = []
        
        for i in range(len(boxes)):
            if IOU_bbox(total_bbox[cur_idx][:4], boxes[i][:4]) > overlapThresh:
                pop_list.append(i)
        boxes = np.delete(boxes, pop_list, 0)
        scores = np.delete(scores, pop_list, 0)
        
        if len(boxes) > 0:
            total_bbox.append(boxes[0]); boxes = np.delete(boxes, 0, 0)
            total_scores.append(scores[0]); scores = np.delete(scores, 0, 0)
            
            cur_idx += 1
    return np.array(total_bbox), np.array(total_scores)

def IOU_bbox(box1, box2):
    """compute overlaps over intersection"""
    ixmin = np.maximum(box2[0], box1[0])
    iymin = np.maximum(box2[1], box1[1])
    ixmax = np.minimum(box2[2], box1[2])
    iymax = np.minimum(box2[3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (box2[2] - box2[0] + 1.) *
           (box2[3] - box2[1] + 1.) - inters)

    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps

window = [-350, 350]
def windowing(im, win = [-350, 350]):
    """scale intensity from win[0]~win[1] to float numbers in 0~255"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

def inter_union_bbox(box1, box2):
    """compute overlaps over intersection"""
    ixmin = np.maximum(box2[0], box1[0])
    iymin = np.maximum(box2[1], box1[1])
    ixmax = np.minimum(box2[2], box1[2])
    iymax = np.minimum(box2[3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (box2[2] - box2[0] + 1.) *
           (box2[3] - box2[1] + 1.) - inters)

    return inters, uni

def miou_list(input_boxes, input_gts, fns):
    inters = []
    unions = []
    fn_list = []
    for input_box, input_gt,fn in zip(input_boxes, input_gts, fns):
        #print(input_box)
        i1 = sorted(input_box, key=lambda x : -x[-1])
        i1_boxes = [i[:4] for i in i1]
        i1_scores = [i[-1] for i in i1]
        nms_box, nms_score = nms_taeu_2(i1_boxes, i1_scores)
        
        for n in nms_box:
            i, u = inter_union_bbox(n, input_gt[0])
            inters.append(i)
            unions.append(u)
            fn_list.append(fn)
    return inters, unions, fn_list

def post_bbox_max_probs(all_boxes, all_scores_logits):
    all_scores_probs = [] # prob after softmax 
    all_scores_max_probs = []
    all_scores_max_label_id = []
    for i in range(len(all_scores_logits)):    
        cur_logit = all_scores_logits[i]
        cur_prob = []
        if len(cur_logit) != 0:
            for j in cur_logit:
                cur_prob.append(softmax(j))
        all_scores_probs.append(np.array(cur_prob))
        mp = []
        mli = []
        a = all_scores_probs[i]
        if len(a) != 0:
            mli = np.argmax(a, axis=1)
            mp = np.max(a, axis = 1)
        all_scores_max_probs.append(np.array(mp))
        all_scores_max_label_id.append(np.array(mli))

    #all_scores_probs
        
    empty_array5 = np.empty((0,5))
    #empty_array5 = empty_array5.reshape((0,5))

    all_boxes_post = []
    all_scores_max_probs_post = []
    all_scores_max_label_id_post = []
    for i in range(len(all_boxes)):
        cur_b = []
        cur_mp = []
        cur_mli = []
        if len(all_boxes[i]) != 0:
            for j in range(len(all_scores_max_label_id[i])):
                if all_scores_max_label_id[i][j] != 0:
                    cur_b.append(np.append(all_boxes[i][j],all_scores_max_probs[i][j]))
                    cur_mp.append(all_scores_max_probs[i][j])
                    cur_mli.append(all_scores_max_label_id[i][j])
        if len(cur_b) == 0:
            cur_b = empty_array5
        else :
            cur_b = np.array(cur_b)
        all_boxes_post.append(cur_b)
        all_scores_max_probs_post.append(np.array(cur_mp))
        all_scores_max_label_id_post.append(np.array(cur_mli))

    return all_boxes_post, all_scores_max_probs_post, all_scores_max_label_id_post

def miou(all_scores_max_label_id_post, 
        all_boxes_post,
        all_gts_label,
        all_gts,
        fns,
        num_classes = 2):
    det_res_list = []
    inters_list = []
    unions_list = []
    fns_list = []
    fps_size = []
    fps_sizes = []
    fps_sizes_fn = []
    i_boxes_list = []
    i_gts_boxes_list = []
    empty_array5 = np.empty((0,5))
    #empty_array5 = empty_array5.reshape((0,5))
    for class_idx in range(1,num_classes+1):

        i_label = [all_scores_max_label_id_post[i] == class_idx for i in range(len(all_scores_max_label_id_post))]
        i_boxes = [all_boxes_post[i][i_label[i]] for i in range(len(all_boxes_post))]
        
        i_gts = [all_gts_label[i] == class_idx for i in range(len(all_gts_label))]
        i_gts_boxes = [all_gts[i][i_gts[i]] for i in range(len(all_gts))]
        
        i_boxes_list.append(i_boxes)
        i_gts_boxes_list.append(i_gts_boxes)

        input_boxes = []
        input_gts_boxes = []
        i_fns = []
        i_fps_boxes = []
        i_fps_fns = []
        for i in range(len(i_gts_boxes)):
            if len(i_gts_boxes[i]) == 0 and len(i_boxes[i])!= 0:
                i1 = sorted(i_boxes[i], key=lambda x : -x[-1])
                i1_boxes = [i[:4] for i in i1]
                i1_scores = [i[-1] for i in i1]
                nms_box, nms_score = nms_taeu_2(i1_boxes, i1_scores)
                nms_new_box = []
                for j in range(len(nms_box)):
                    if nms_score[j] > 0.4:
                        nms_new_box.append(nms_box[j])
                i_fps_boxes.append(nms_new_box)
                i_fps_fns.append(fns[i])
            if len(i_gts_boxes[i]) == 0 :
                continue
            input_boxes.append(i_boxes[i])
            input_gts_boxes.append(i_gts_boxes[i])
            i_fns.append(fns[i])
            
        for i in range(len(i_fns)):
            if 'B' in i_fns[i]:
                i_fns[i] = i_fns[i][:9]
            else:
                i_fns[i] = i_fns[i][:8]
        i_fps_boxes_left = []
        i_fps_boxes_right = []
        i_fps_fns_left = []
        i_fps_fns_right = []
        
        for cur_bbox, fn in zip(i_fps_boxes, i_fps_fns):
            for j in cur_bbox:
                if j[1] < 256:
                    i_fps_boxes_left.append(j)
                    i_fps_fns_left.append(fn)
                else:
                    i_fps_boxes_right.append(j)
                    i_fps_fns_right.append(fn)
        
        fps_sizes_left = []
        fps_sizes_right = []
        for box1 in i_fps_boxes_left:
            fps_sizes_left.append((box1[2] - box1[0]) * (box1[3] - box1[1]))
        for box1 in i_fps_boxes_right:
            fps_sizes_right.append((box1[2] - box1[0]) * (box1[3] - box1[1]))
        fps_sizes.append(fps_sizes_left)
        fps_sizes.append(fps_sizes_right)
        fps_sizes_fn.append(i_fps_fns_left)
        fps_sizes_fn.append(i_fps_fns_right)
        fps_unions_left = np.sum(fps_sizes_left)
        fps_unions_right = np.sum(fps_sizes_right)
        fps_unions_all = fps_unions_left + fps_unions_right

        input_boxes_left = []
        input_boxes_right = []
        input_gts_boxes_left = []
        input_gts_boxes_right = []
        i_fns_left = []
        i_fns_right = []
        if num_classes == 2:
            for cur_bbox, cur_label, cur_fn in zip(input_boxes, input_gts_boxes, i_fns):
                cur_left = []
                cur_right = []
                for j in cur_bbox:
                    if j[1] < 256:
                        cur_left.append(j)

                    else:
                        cur_right.append(j)
                if len(cur_left) == 0:
                    cur_left = empty_array5
                if len(cur_right) == 0:
                    cur_right = empty_array5

                if len(cur_label) == 1:
                    if cur_label[0][1] < 256:
                        left_check = True
                        input_gts_boxes_left.append(cur_label)
                        input_boxes_left.append(cur_left)
                        i_fns_left.append(cur_fn)
                    else:
                        input_gts_boxes_right.append(cur_label)
                        input_boxes_right.append(cur_right)
                        i_fns_right.append(cur_fn)
                else:
                    input_gts_boxes_left.append(np.array([cur_label[0]]))
                    input_boxes_left.append(cur_left)
                    i_fns_left.append(cur_fn)
                    input_gts_boxes_right.append(np.array([cur_label[1]]))
                    input_boxes_right.append(cur_right)
                    i_fns_right.append(cur_fn)
        if len(input_gts_boxes) != 0 and len(input_boxes) != 0:
            print('len input_boxes : ',len(input_boxes), len(input_gts_boxes))

            if num_classes == 2 :
                det_res = sens_at_FP(input_boxes_left, input_gts_boxes_left, cfg.TEST.VAL_FROC_FP, cfg.TEST.IOU_TH) # i_gts_boxes 에는 적어도 label 이 잇는 것들만 담아야겠네
                det_res_list.append(det_res)
                det_res = sens_at_FP(input_boxes_right, input_gts_boxes_right, cfg.TEST.VAL_FROC_FP, cfg.TEST.IOU_TH) # i_gts_boxes 에는 적어도 label 이 잇는 것들만 담아야겠네

                det_res_list.append(det_res)
                print('class idx : ',class_idx)
                inters ,unions, fn_list = miou_list(input_boxes_left, input_gts_boxes_left, i_fns_left)
                print(np.sum(inters) / (np.sum(unions)+fps_unions_left))
                fps_size.append(fps_unions_left)
                inters_list.append(inters); unions_list.append(unions); fns_list.append(fn_list)
                inters ,unions, fn_list = miou_list(input_boxes_right, input_gts_boxes_right, i_fns_right)
                print(np.sum(inters) / (np.sum(unions)+fps_unions_right))
                fps_size.append(fps_unions_right)
                inters_list.append(inters); unions_list.append(unions); fns_list.append(fn_list)
            else:
                det_res = sens_at_FP(input_boxes, input_gts_boxes, cfg.TEST.VAL_FROC_FP, cfg.TEST.IOU_TH)
                det_res_list.append(det_res)
                print('class idx : ', class_idx)
                inters, unions, fn_list = miou_list(input_boxes, input_gts_boxes, i_fns)
                print(np.sum(inters) / (np.sum(unions) + fps_unions_all))
                fps_size.append(fps_unions_all)
                inters_list.append(inters);
                unions_list.append(unions);
                fns_list.append(fn_list)

    return inters_list, unions_list, fps_size, i_boxes_list, i_gts_boxes_list
    
def roc_auc(boxes_all,gts_all, version = 'a',save_dir = './ag/result_slice/'):
    avgFP = [0.5,1,2,4]
    iou_th = 0.5
    nImg = len(boxes_all)
    print('nImg : ',nImg)
    img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
    print('len img_idxs : ',len(img_idxs)) # bbx , gts pair 만큼 늘려주는거
    bbx_num_list = [0 for i in range(nImg)]
    print('len bbx num list : ',len(bbx_num_list))
    for i in img_idxs:
        bbx_num_list[i] += 1
    print('bbx_num_list[:10] : ', bbx_num_list[:10])
    print('avg bbx_num_list : ',np.mean(bbx_num_list))
    boxes_cat = np.vstack(boxes_all)
    print('boxes_cat shape : ',boxes_cat.shape) # 마지막 인덱스 기준으로 쭉 피는 거 인듯?
    scores = boxes_cat[:,-1]
    ord = np.argsort(scores)[::-1] # 역순
    scores_ordered = scores[ord]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs[ord]

    hits = [np.zeros((len(gts),), dtype = bool) for gts in gts_all]
    not_hits = [np.ones((len(gts),), dtype = bool) for gts in gts_all]
    not_count = [idx for idx in range(len(gts_all)) if np.sum(gts_all[idx]) == 0 ]
    print(len(hits), len(hits[0]))
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    fns = []
    truths = []
    ssss = []
    two_nodule = 0
    for i in gts_all:
        if np.sum(i) == 0:
            continue
        else:
            if len(i) == 2:
                two_nodule += 1
    print(two_nodule)
    print('not count : ', len(not_count))

    Precision = []
    Recall = []
    TP = FP = 0
    #FN = len(hits) - len(not_count) + two_nodule
    nGt = len(hits) - len(not_count) + two_nodule
    ssss = []
    for i in tqdm(range(len(boxes_cat))):
        if img_idxs[i] not in not_count:
            overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
            if len(overlaps) == 0 or overlaps.max() < iou_th :
                FP += 1
                ssss.append(scores_ordered[i])
            else :
                for j in range(len(overlaps)):
                    if overlaps[j] >= iou_th and not hits[img_idxs[i]][j] :
                        TP += 1
                        #print('img_idx append : ',img_idxs[i], 'with this prob : ',scores_ordered[i])
                        hits[img_idxs[i]][j] = True
                        ssss.append(scores_ordered[i])
            
        else :
            
            FP += 1
            ssss.append(scores_ordered[i])
            
        try : 
            AP = TP / (TP + FP)
            Rec = TP / (nGt)

        except:
            AP = Rec = 0.0

        Precision.append(AP)
        Recall.append(Rec)

    area_rp = auc(Recall, Precision)
    plt.plot(Recall, Precision, lw = 2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    # plt.xlim([0, 1.0])
    # plt.ylim([0,1.05])
    plt.grid(color='#CCCCCC', linestyle='--', linewidth=1, )
    plt.savefig(save_dir+ version + '.png', dpi=300)
    plt.show()
    return area_rp

def i_boxes_left_right(i_boxes, i_gts_boxes):
    i_gts_boxes_left = []
    i_gts_boxes_right = []
    i_boxes_right = []
    i_boxes_left = []
    cnt = 0
    b_cnt = 0
    empty_array5 = np.empty((0, 5))
    empty_4 = np.empty((0,4))
    for i in range(len(i_gts_boxes)):
        if len(i_gts_boxes[i]) == 0 or np.sum(i_gts_boxes[i]) == 0 :
            i_gts_boxes_left.append(i_gts_boxes[i])
            i_gts_boxes_right.append(i_gts_boxes[i])
            cnt += 1
        else: 
            b_cnt +=1
            cur_label = i_gts_boxes[i]
            if len(cur_label) == 1:
                if cur_label[0][1] < 256:
                    i_gts_boxes_left.append(cur_label)
                    i_gts_boxes_right.append(empty_4)
                else:
                    i_gts_boxes_left.append(empty_4)
                    i_gts_boxes_right.append(cur_label)
                #print(cur_label)
            elif len(cur_label) == 2:
                print(2)
                i_gts_boxes_left.append(np.array([cur_label[0]]))
                i_gts_boxes_right.append(np.array([cur_label[1]]))
            else:
                print(cur_label)
        
        
        cur_left = []
        cur_right = []
        for j in i_boxes[i]:
            if j[1] < 256:
                cur_left.append(j)
            else:
                cur_right.append(j)
        if len(cur_left) == 0:
            cur_left = empty_array5
        if len(cur_right) == 0:
            cur_right = empty_array5
        i_boxes_left.append(cur_left)
        i_boxes_right.append(cur_right)
    return i_gts_boxes_left,i_gts_boxes_right,i_boxes_left,i_boxes_right

def make_post_df(fns, all_gts, all_gts_label, all_boxes_post, all_scores_max_label_id_post, all_scores_max_probs_post):
    all_gts_label_list = [list(i) for i in all_gts_label]
    all_labels_list = [list(i) for i in all_scores_max_label_id_post]
    all_gts_list = [list(i) for i in all_gts]
    for i in range(len(all_gts_list)):
        for j in range(len(all_gts_list[i])):
            if np.sum(all_gts_list[i][j]) == 0:
                all_gts_list[i] = []
            else:
                all_gts_list[i][j] = list(all_gts_list[i][j])
    all_scores_list = [list(i) for i in all_scores_max_probs_post]

    for i in range(len(all_scores_list)):
        if len(all_scores_list[i]) != 0:
            for j in range(len(all_scores_list[i])):
                all_scores_list[i][j] = np.round(all_scores_list[i][j],3)
                
    all_boxes_post_list = [list(i) for i in all_boxes_post]
    for i in range(len(all_boxes_post_list)):
        if len(all_boxes_post_list[i]) != 0:
            for j in range(len(all_boxes_post_list[i])):
                all_boxes_post_list[i][j] = list(all_boxes_post_list[i][j])
                for k in range(len(all_boxes_post_list[i][j])):
                    if k != 4:
                        all_boxes_post_list[i][j][k] = int(all_boxes_post_list[i][j][k])
                    else : 
                        all_boxes_post_list[i][j][k] = np.round(all_boxes_post_list[i][j][k], 3)
        
    fns_name= []; fns_keys = []
    for fn in fns:
        if 'B' in fn :
            fns_name.append(fn[:9])
            fns_keys.append(fn[10:])
        else:
            fns_name.append(fn[:8])
            fns_keys.append(fn[9:])
            
    for i in range(len(fns_keys)):
        cur = fns_keys[i]
        if len(cur) == 1:
            cur = '00' + cur
        elif len(cur) == 2:
            cur = '0' + cur
        fns_keys[i] = cur


    for i in range(len(all_labels_list)):
        if len(all_labels_list[i]) != 0:
            total_list = []
            for j in range(len(all_labels_list[i])):
                total_list.append((all_labels_list[i][j], all_scores_list[i][j], all_boxes_post_list[i][j]))
            total_list = sorted(total_list, key=lambda x: -x[1])
            for j in range(len(total_list)):
                all_labels_list[i][j] = total_list[j][0]
                all_scores_list[i][j] = total_list[j][1]
                all_boxes_post_list[i][j] = total_list[j][2]
                
    d = {'File_name': fns_name, 'index':fns_keys, 'gts_label':all_gts_label_list, 'predict_label':all_labels_list,  'predict_scores':all_scores_list, 'gts_bbox' : all_gts_list ,'predict_bbox':all_boxes_post_list}
    df2 = pd.DataFrame(data=d)
    df2 = df2.sort_values(by=['File_name', 'index'])
    df2 = df2.reset_index(drop=True)
    return df2

def fn_feature_dict_from_df(df2, num_classes):
    fn_label_dict = {}
    for index, row in df2.iterrows():
        fn = row['File_name']
        if fn not in fn_label_dict:
            fn_label_dict[fn] = 0
        gts_label = row['gts_label']
        if num_classes == 2:
            if 2 in gts_label:
                fn_label_dict[fn] = 1
        elif num_classes == 4:
            if 3 in gts_label or 4 in gts_label:
                fn_label_dict[fn] = 1
        elif num_classes == 1:
            if 1 in gts_label:
                fn_label_dict[fn]= 1

    fn_list =list(fn_label_dict.keys())
    fn_feature_dict = {}
    for idx in tqdm(range(len(fn_list))):
        cur_df = df2[df2['File_name'] == fn_list[idx]]
        cur_max_len = len(cur_df)
        index_list = cur_df['index'].tolist()
        predict_label_list = cur_df['predict_label'].tolist()
        predict_scores_list = cur_df['predict_scores'].tolist()
        predict_bbox_list = cur_df['predict_bbox'].tolist()
        new_idx  = []
        new_scores = []
        new_bbox = []
        new_bbox_len = []

        for i in range(len(predict_label_list)):
            check = False

            if len(predict_label_list[i]) != 0:
                tmp_scores = []
                tmp_bbox = []
                for j in range(len(predict_label_list[i])):
                    if num_classes == 2:
                        if predict_label_list[i][j] == 2:
                            check = True
                            tmp_scores.append(predict_scores_list[i][j])
                            tmp_bbox.append(predict_bbox_list[i][j])
                    elif num_classes == 4:
                        if predict_label_list[i][j] > 2:
                            check = True
                            tmp_scores.append(predict_scores_list[i][j])
                            tmp_bbox.append(predict_bbox_list[i][j])
                    elif num_classes == 1:
                        if predict_label_list[i][j] == 1:
                            check = True
                            tmp_scores.append(predict_scores_list[i][j])
                            tmp_bbox.append(predict_bbox_list[i][j])
                cur_max = 0
                cur_bbox_len = len(tmp_scores)
                for k,b in zip(tmp_scores, tmp_bbox):
                    if cur_max < k:
                        cur_max = k
                        max_bbox = b
                        
            if check:
                new_idx.append(int(index_list[i]))
                new_scores.append(cur_max)
                new_bbox.append(max_bbox)
                new_bbox_len.append(cur_bbox_len)


        intv_max = []
        intv_len = []
        begin_k = 0
        if len(new_idx) > 0:
            for k in range(len(new_idx)-1):
                if new_idx[k+1] - new_idx[k] < 3:
                    continue

                if k+1 - begin_k >= 2:
                    intv_max.append(max(new_scores[begin_k:k+1]))
                    intv_len.append(k+1 - begin_k)
                begin_k = k+1
        if len(new_scores) > 1:
            if k+1 - begin_k >= 2:
                intv_max.append(max(new_scores[begin_k:]))
                intv_len.append(k+2 - begin_k)
        cur_dict = {}
        cur_dict['intv_max'] = intv_max
        cur_dict['intv_len'] = intv_len
        if len(new_scores) == 0:
            cur_dict['max_one'] = 0.0
            cur_dict['max_slice'] = 0
        else:
            cur_dict['max_one'] = max(new_scores)
            cur_dict['max_slice'] = new_idx[np.argmax(new_scores)]
        cur_dict['new_scores'] = new_scores
        cur_dict['new_bbox'] = new_bbox
        cur_dict['new_bbox_len'] = new_bbox_len
        cur_dict['new_idx'] = new_idx
        fn_feature_dict[fn_list[idx]] = cur_dict


    return fn_feature_dict, fn_label_dict

def plt_roc(fpr, tpr, roc_auc):
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

def ap_(label, predict, logger):

    ap_score = average_precision_score(label, predict) #
    logger.info('ap score : {}'.format(ap_score))
    roc_auc = roc_auc_score(label, predict) #
    logger.info('roc auc score : {}'.format(roc_auc))

    fpr, tpr, threshold = metrics.roc_curve(label, predict) #

    

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    post_predict = [] 
    for i in predict: # 
        if i >= optimal_threshold:
            post_predict.append(1)
        else:
            post_predict.append(0)
    accuracy_score(label, post_predict)
    logger.info('optimal acc : {}'.format(accuracy_score(label, post_predict)))
    logger.info('optimal tsh : {}'.format(optimal_threshold))

    label0 = []
    label1 = []
    post0 = []
    post1 = []
    for i in range(len(label)):
        if label[i] == 0:
            label0.append(label[i])
            post0.append(post_predict[i])
        else:
            label1.append(label[i])
            post1.append(post_predict[i])
    logger.info('label0 case : {}, acc : {}'.format(len(label0), accuracy_score(label0, post0)))
    logger.info('label1 case : {}, acc : {}'.format(len(label1), accuracy_score(label1, post1)))


def make_result_df(fn_feature_dict,):
    # 이거는 조금있다가 만들자. result_val_0921.pkl 에 관련된 정보 있을테니까
    # prediction ~ left-right 폴더보고 다시 만들기
    fn_list = []
    intv_max = []
    intv_len = []
    max_score = []
    max_slice_idx = []
    max_score_of_the_slice = []
    bbox_len_of_the_slice = []
    bbox_of_the_slice = []
    bbox_slice_idx = []
    gts = []
    pred = []
    pred_score = []
    correct = []

    for key, value in fn_feature_dict.items():
        fn_list.append(key)
        intv_max.append(value['intv_max']); intv_len.append(value['intv_len']); max_score.append(value['max_one']);
        max_slice_idx.append(value['max_slice']); max_score_of_the_slice.append(value['new_scores']);
        bbox_len_of_the_slice.append(value['new_bbox_len']); bbox_of_the_slice.append(value['new_bbox']);
        bbox_slice_idx.append(value['new_idx']); gts.append(value['gts']); pred.append(value['pred'])
        pred_score.append(value['pred_score']); correct.append(value['correct'])

    df_val = pd.read_pickle('AG_AGN2')
    df_val = df_val[df_val['Train_val_test'] != 1]


def fn_feature_dict_from_df_eachAGN(df2, num_classes, label_id):
    fn_label_dict = {}
    for index, row in df2.iterrows():
        fn = row['File_name']
        if fn not in fn_label_dict:
            fn_label_dict[fn] = 0
        gts_label = row['gts_label']
        if label_id in gts_label:
            fn_label_dict[fn] = 1

    fn_list = list(fn_label_dict.keys())
    fn_feature_dict = {}
    for idx in tqdm(range(len(fn_list))):
        cur_df = df2[df2['File_name'] == fn_list[idx]]
        cur_max_len = len(cur_df)
        index_list = cur_df['index'].tolist()
        predict_label_list = cur_df['predict_label'].tolist()
        predict_scores_list = cur_df['predict_scores'].tolist()
        predict_bbox_list = cur_df['predict_bbox'].tolist()
        new_idx = []
        new_scores = []
        new_bbox = []
        new_bbox_len = []

        for i in range(len(predict_label_list)):
            check = False

            if len(predict_label_list[i]) != 0:
                tmp_scores = []
                tmp_bbox = []
                for j in range(len(predict_label_list[i])):
                    if predict_label_list[i][j] == label_id:
                        check = True
                        tmp_scores.append(predict_scores_list[i][j])
                        tmp_bbox.append(predict_bbox_list[i][j])
                cur_max = 0
                cur_bbox_len = len(tmp_scores)
                for k, b in zip(tmp_scores, tmp_bbox):
                    if cur_max < k:
                        cur_max = k
                        max_bbox = b

            if check:
                new_idx.append(int(index_list[i]))
                new_scores.append(cur_max)
                new_bbox.append(max_bbox)
                new_bbox_len.append(cur_bbox_len)

        intv_max = []
        intv_len = []
        begin_k = 0
        if len(new_idx) > 0:
            for k in range(len(new_idx) - 1):
                if new_idx[k + 1] - new_idx[k] < 3:
                    continue

                if k + 1 - begin_k >= 2:
                    intv_max.append(max(new_scores[begin_k:k + 1]))
                    intv_len.append(k + 1 - begin_k)
                begin_k = k + 1
        if len(new_scores) > 1:
            if k + 1 - begin_k >= 2:
                intv_max.append(max(new_scores[begin_k:]))
                intv_len.append(k + 2 - begin_k)
        cur_dict = {}
        cur_dict['intv_max'] = intv_max
        cur_dict['intv_len'] = intv_len
        if len(new_scores) == 0:
            cur_dict['max_one'] = 0.0
            cur_dict['max_slice'] = 0
        else:
            cur_dict['max_one'] = max(new_scores)
            cur_dict['max_slice'] = new_idx[np.argmax(new_scores)]
        cur_dict['new_scores'] = new_scores
        cur_dict['new_bbox'] = new_bbox
        cur_dict['new_bbox_len'] = new_bbox_len
        cur_dict['new_idx'] = new_idx
        fn_feature_dict[fn_list[idx]] = cur_dict

    return fn_feature_dict, fn_label_dict