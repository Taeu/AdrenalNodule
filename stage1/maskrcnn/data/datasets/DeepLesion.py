# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""The DeepLesion dataset loader, include box, tag, and masks"""
import torch
import torchvision
import numpy as np
import os
import csv
import logging
import json

from maskrcnn.data.datasets.load_ct_img import load_prep_img, load_prep_mask, mask_to_xylist
from maskrcnn.structures.bounding_box import BoxList
from maskrcnn.structures.segmentation_mask import SegmentationMask
from maskrcnn.config import cfg
from maskrcnn.data.datasets.DeepLesion_utils import load_tag_dict_from_xlsfile, gen_mask_polygon_from_recist, load_lesion_tags
from maskrcnn.data.datasets.DeepLesion_utils import gen_parent_list, gen_exclusive_list, gen_children_list


class DeepLesionDataset(object):

    def __init__(
        self, split, data_dir, ann_file, transforms=None
    ):
        self.transforms = transforms
        self.split = split
        self.data_path = data_dir
        self.classes = ['__background__',  # always index 0
                        #'AGR','AGL',
                        'AGNR','AGNL'] ################## dataloader 수정1
        self.num_classes = len(self.classes)
        self.loadinfo(ann_file) # 이부분 수정중 
        
        print('len(self.filenames)', len(self.filenames))
        self.image_fn_list, self.lesion_idx_list = self.load_split_index()
        print('len(self.image_fn_list)', len(self.image_fn_list))
        self.num_images = len(self.image_fn_list)
        self.logger = logging.getLogger(__name__)
        self.logger.info('AGN datasets %s num_images: %d' % (split, self.num_images))

    def _process_manual_annot_test_tags(self):
        pass
        
    def _process_tags(self):
        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, info).
        """
        idx = self.lesion_idx_list[index]
        image_fn = self.filenames[idx]
        slice_idx = self.slice_idx[idx]
        nodule_size = self.nodule_size[idx]
        slice_intv = self.slice_intv[idx] 
        spacing = self.spacing[idx] 
        num_slice = cfg.INPUT.NUM_SLICES * cfg.INPUT.NUM_IMAGES_3DCE
        is_train = self.split=='train'
        cfg.INPUT.IMG_DO_CLIP = False
        if is_train and cfg.INPUT.DATA_AUG_3D is not False:
            pass
        im= load_prep_img(self.data_path, image_fn, spacing, slice_intv, slice_idx,
                                           cfg.INPUT.IMG_DO_CLIP, num_slice=num_slice, is_train=is_train)
        #im -= cfg.INPUT.PIXEL_MEAN
        im /= 255.
        im = torch.from_numpy(im.transpose((2, 0, 1))).to(dtype=torch.float)

        ## add bbox, mask, classes to BoxList from binary mask
        boxes = self.bbox_list[idx]
        classes = self.label_id[idx]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (im.shape[2], im.shape[1]), mode="xyxy")
        num_boxes = boxes.shape[0]
        classes = torch.as_tensor(classes, dtype=torch.int)  # lesion/nonlesion
        target.add_field("labels", classes)
        
        if cfg.MODEL.MASK_ON:
            target.add_field("masks", binary_masks)
            
        if self.transforms is not None:
            im, target = self.transforms(im, target)

        hflip = False
        vflip = False
        
        if self.split == 'train': # data augmentation
            hflip = np.random.random() < 0.5
            vflip = np.random.random() < 0.5

            translation = int(np.random.uniform(-30, 30))
            im, target.bbox = translation_img_bbox(im, target.bbox, translation)

            if hflip:
                for i in range(im.shape[0]):
                    im[i,:,:] = torch.from_numpy(np.fliplr(im[i].numpy()).copy())
                target.bbox = bbox_fliplr(target.bbox, im.shape[1:])

            if vflip:
                for i in range(im.shape[0]):
                    im[i] = torch.from_numpy(np.flipud(im[i].numpy()).copy())
                target.bbox = bbox_flipud(target.bbox, im.shape[1:])
                                      
        infos = {'im_index': index, 'slice_index': slice_idx, 'image_fn': image_fn, 
                'spacing': spacing, 'slice_intv': slice_intv, 'nodule_size' : nodule_size,
                'vflip' : vflip, 'hflip': hflip}
        
        return im, target, infos

    def __len__(self):
        return len(self.image_fn_list)

    def load_split_index(self):
        """
        need to group lesion indices to image indices, since one image can have multiple lesions
        :return:
        """

        split_list = ['train', 'val', 'test', 'small']
        index = split_list.index(self.split)
        if self.split != 'small':
            lesion_idx_list = np.where((self.train_val_test == index + 1))[0]
        else:
            lesion_idx_list = np.arange(30)
        
        fn_list = self.filenames[lesion_idx_list]
        return fn_list, lesion_idx_list#fn_list_unique, lesion_idx_grouped

    def loadinfo(self, path):
        """load annotations and meta-info from DL_info.csv"""
        info = []
        import pandas as pd
        #df = pd.read_csv(path,index_col=0)
        df = pd.read_pickle(path)
        self.filenames = df['File_name'].values
        self.slice_idx = df['Key_slice_index'].values
        self.nodule_size = df['size'].values
        class_idx_list = [0,1,2,1,2]# [0,1,2,3,4] # ################## dataloader 수정1
        llid = []
        for li in df['label_id'].values:
            cur_li = []
            for i in li:
                cur_li.append(class_idx_list[i])
            llid.append(cur_li)
        self.label_id = llid
        self.train_val_test = df['Train_val_test'].values
        self.spacing = df['Spacing'].values
        self.slice_intv = df['Slice_intv'].values
        self.bbox_list = df['bbox_list'].values
        
        


def translation_img_bbox(im, bbox_list, translation):
    
    for i in range(im.shape[0]) :
        img_shape = im.shape[1]
        if translation >= 0 :
            copy_map = torch.from_numpy(im[i][:img_shape-translation, :img_shape-translation].numpy().copy())
            im[i][translation:,translation:] = copy_map
        else:
            copy_map = torch.from_numpy(im[i][-translation:, -translation:].numpy().copy())
            im[i][:img_shape+translation, :img_shape+translation] = copy_map
    new_bbox = []
    for bbox in bbox_list:
        x1,y1,x2,y2 = bbox
        new_bbox.append([x1+translation, y1+translation, x2+translation, y2+translation])
    
    return im, torch.as_tensor(new_bbox)

def bbox_flipud(bbox_list, img_size):
    new_bbox = []
    x_img_size = img_size[0]
    
    for bbox in bbox_list:
        
        x1,y1,x2,y2 = bbox
        new_x1 = x_img_size - x2
        new_x2 = x_img_size - x1
        new_bbox .append([new_x1, y1, new_x2, y2])
        
   
    return torch.as_tensor(new_bbox)

def bbox_fliplr(bbox_list, img_size):
    new_bbox = []
    y_img_size = img_size[1]
    
    for bbox in bbox_list:
        x1,y1,x2,y2 = bbox


        new_y1 = y_img_size - y2
        new_y2 = y_img_size - y1

        new_bbox.append([x1, new_y1, x2, new_y2])
    return torch.as_tensor(new_bbox)      
        
def img_bbox_resize(self, img, bbox_list, to_img_size):
    from skimage.transform import resize
    
    bottle_resized = resize(im[i], to_img_size)
    
    from_img_size = img.shape[1:]
    for i in range(im.shape[0]):
        img[i] = torch.from_numpy(resize(im[i].numpy(), to_img_size).copy())

    #scale label
    width_scale_factor = to_img_size[0] / from_img_size[0] 
    height_scale_factor = to_img_size[1] / from_img_size[1]

    new_bbox_list = []
    for label in bbox_list:
        #x, w 
        label[0] *= width_scale_factor
        label[2] *= width_scale_factor

        #y, h
        label[1] *= height_scale_factor
        label[3] *= height_scale_factor
        new_bbox_list.append(label)

    return img,  torch.as_tensor(new_bbox_list)