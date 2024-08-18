# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""Evaluation code of the DeepLesion dataset"""
import logging
import numpy as np
from scipy.spatial.distance import cdist
import torch

from .detection_eval import sens_at_FP
from .tagging_eval import compute_all_acc_wt, compute_thresholds, print_accs
from maskrcnn.config import cfg
from maskrcnn.utils.print_info import save_acc_to_file


def do_evaluation(
    dataset,
    predictions,
    is_validation
):
    logger = logging.getLogger("maskrcnn.inference")
    print(len(dataset), len(predictions))
    #assert len(dataset) == len(predictions)

    fns = list(predictions.keys())
    if 'is_gt' in predictions[fns[0]]['result'].extra_fields.keys(): # pass
        for i, fn in enumerate(fns):
            is_gt = predictions[fn]['result'].get_field('is_gt')
            predictions[fn]['gt_result'] = predictions[fn]['result'][is_gt]
            predictions[fn]['result'] = predictions[fn]['result'][~is_gt]

    # lesion detection (lesion vs. non-lesion)
    det_res = eval_DL_detection(predictions, logger, is_validation)

    # weakly-supervised segmentation
    """
    if cfg.MODEL.MASK_ON:
        logger.info('\nSegmentation accuracy:')
        seg_res = eval_DL_segmentation(predictions, logger)
    """
    return np.mean(det_res[:4]) # 


def eval_DL_detection(predictions, logger, is_validation):
    fns = sorted(predictions.keys())
    all_boxes = [predictions[fn]['result'].bbox.numpy() for fn in fns]
    all_scores = [predictions[fn]['result'].get_field('scores').numpy() for fn in fns]
    all_labels = [predictions[fn]['result'].get_field('labels').numpy() for fn in fns]
    all_gts = [predictions[fn]['target'].bbox.cpu().numpy() for fn in fns]
    all_gts_label = [predictions[fn]['target'].get_field('labels').cpu().numpy() for fn in fns]

    det_res_list = [] 
    for class_idx in range(1, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES):
        i_label = [all_labels[i] == class_idx for i in range(len(all_labels))]
        i_boxes = [all_boxes[i][i_label[i]] for i in range(len(all_boxes))]
        i_scores = [all_scores[i][i_label[i]] for i in range(len(all_scores))]
        i_boxes = [np.hstack((b, s.reshape((-1, 1)))) for (b, s) in zip(i_boxes, i_scores)]

        i_gts = [all_gts_label[i] == class_idx for i in range(len(all_gts_label))]
        i_gts_boxes = [all_gts[i][i_gts[i]] for i in range(len(all_gts))]

        logger.info('\nDetection accuracy:')
        logger.info('Sensitivity @ %s average FPs per image:', str(cfg.TEST.VAL_FROC_FP))
        
        input_boxes = []
        input_gts_boxes = []
        for i in range(len(i_gts_boxes)):
            if len(i_gts_boxes[i]) == 0 :
                continue
            input_boxes.append(i_boxes[i])
            input_gts_boxes.append(i_gts_boxes[i])

        if len(input_gts_boxes) != 0 and len(input_boxes) != 0:
            print('len input_boxes : ',len(input_boxes), len(input_gts_boxes))
            det_res = sens_at_FP(input_boxes, input_gts_boxes, cfg.TEST.VAL_FROC_FP, cfg.TEST.IOU_TH) # i_gts_boxes 에는 적어도 label 이 잇는 것들만 담아야겠네
            det_res_list.append(det_res)
            print('class idx : ',class_idx)
            logger.info(', '.join(['%.4f'%v for v in det_res]))
            logger.info('mean of %s: %.4f', str(cfg.TEST.VAL_FROC_FP[:4]), np.mean(det_res[:4]))

    return det_res

def eval_DL_tagging(predictions, logger, is_validation):
    logger.info('\nTagging accuracy:')

    fns = sorted(predictions.keys())
    prob_all = [predictions[fn]['gt_result'].get_field('tag_scores') for fn in fns]
    pred_all = [predictions[fn]['gt_result'].get_field('tag_predictions') for fn in fns]
    prob_all = torch.cat(prob_all).cpu().numpy()
    pred_all = torch.cat(pred_all).cpu().numpy()

    if is_validation:  # compute best class thresholds for next test
        target_all = [predictions[fn]['target'].get_field('tags') for fn in fns]
        target_all = torch.cat(target_all).cpu().numpy()
        pred_wt_all = target_all >= 0
        logger.info('mined tags from reports:')
        accs = compute_all_acc_wt(target_all, pred_all, prob_all, pred_wt_all)
        print_accs(accs, logger)
        save_acc_to_file(accs, 'val_mined')

        if cfg.TEST.TAG.CALIBRATE_TH:
            tag_sel_val = compute_thresholds(target_all, prob_all, pred_wt_all)
            cfg.runtime_info.tag_sel_val = torch.from_numpy(tag_sel_val).to(torch.float)

    else:  # doing final test using manual annotations
        # in the test set, only 500 hand-labeled tags are released
        assert 'manual_annot_test_tags' in predictions[fns[0]]['target'].extra_fields.keys(), \
            "Currently we evaluate tagging accuracy on manual_annot_test_tags, which is only in the test set of DeepLesion"
        target_all = [predictions[fn]['target'].get_field('manual_annot_test_tags') for fn in fns]
        target_all = torch.cat(target_all).cpu().numpy()
        pred_wt_all = target_all >= 0
        accs = compute_all_acc_wt(target_all, pred_all, prob_all, pred_wt_all)
        save_acc_to_file(accs, 'test_handlabeled')
        logger.info('hand-labeled tags:')
        print_accs(accs, logger)

    return accs['mean_auc']


def eval_DL_segmentation(predictions, logger):
    min_dists = []
    diam_errs = []
    fns = sorted(predictions.keys())
    for fn in fns:
        d = predictions[fn]
        # coordinate offset caused by IMG_DO_CLIP is not considered
        spacing = d['info']['spacing']
        im_scale = d['info']['im_scale']
        gt_recists = d['info']['recists']
        gt_recists_mm = gt_recists/im_scale*spacing

        if not cfg.TEST.EVAL_SEG_TAG_ON_GT:
            raise NotImplementedError
        contours = d['gt_result'].get_field('contour_mm')
        predicted_diameters = d['gt_result'].get_field('diameter_mm')
        for gt_idx in range(len(gt_recists)):
            contour = contours[gt_idx][contours[gt_idx][:,0]>0, :]
            gt_recist = gt_recists_mm[gt_idx]
            min_dists.append(compute_recist_contour_dist(contour, gt_recist))

            # compute the error of lesion diameter estimation
            predicted_diameter = predicted_diameters[gt_idx]
            diam_errs.append(compute_diameter_error(predicted_diameter, gt_recist))

    # print(min_dists)
    # print(diam_errs)
    min_dists_avg = np.mean(min_dists)
    diam_errs_avg = np.mean(diam_errs)
    logger.info('avg min distance (mm) from groundtruth recist points to predicted contours in GT boxes:\n'
                'error of lesion diameter (mm) estimated from predicted contours in GT boxes:\n'
                '%.4f+-%.4f, %.4f+-%.4f',
                min_dists_avg, np.std(min_dists), diam_errs_avg, np.std(diam_errs))
    # print(np.sort(diam_errs)[::10])
    return np.mean([min_dists_avg, diam_errs_avg])


def compute_recist_contour_dist(contour, recist):
    """Avg min distance (mm) from groundtruth recist points to predicted contours"""
    recist = recist.reshape((4, 2))
    D = cdist(recist, contour)
    d4 = D.min(axis=1)
    return d4.mean()


def compute_diameter_error(predicted_diameter, gt_recist):
    """Error of lesion diameter (mm) estimated from predicted contours"""
    gt_recist = gt_recist.reshape((4,2))
    gt_diameters = np.sqrt(np.sum((gt_recist[::2] - gt_recist[1::2]) ** 2, axis=1))
    error = np.mean([np.abs(gt_diameters.max() - predicted_diameter.max()),
                     np.abs(gt_diameters.min() - predicted_diameter.min())])
    return error



