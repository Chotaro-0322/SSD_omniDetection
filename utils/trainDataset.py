"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os
import os.path as osp
import sys

#from utils.augmentations import SSDAugmentation
import cv2
import numpy as np
import torch
import torch.utils.data as data

from config import HOME, CLASSES

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

#VOTT_CLASSES = ('person', ' ')
# note: if you used our download scripts, this should be right
#VOTT_ROOT = osp.join(HOME, "data/VOTT/")


class AnnotationTransform(object):
    """Transforms a annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES, range(len(CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class trainDataset(data.Dataset):
    """Detection Dataset Object

    input is image, target is annotation

    Arguments:
    """

    def __init__(self, root,
                 transform=None,
                 target_transform=AnnotationTransform(),
                 ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join("%s", 'Annotations', '%s.xml')
        self._imgpath = osp.join("%s", 'JPEGImages', '%s.jpg')
        self.ids = list()

        rootpath = osp.join(self.root)
        for line in open(osp.join(rootpath, 'ImageSets', 'Main', 'person_train' + '.txt')):
            self.ids.append((rootpath, line.strip().strip('.jpg')))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        #print("img_id is ", img_id)
        #print("self._annopath is ", self._annopath)
        target = ET.parse(self._annopath % img_id).getroot()
        #print("target is ", self._annopath % img_id)
        img = cv2.imread(self._imgpath % img_id)

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            #print("img is ", img.shape)
            #check_image(img, boxes)
            img = img[:, :, (2, 1, 0)]
            img = img.astype(np.float32)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def check_image(img, boxes):
    imgCV = img
    print("img is ", img.shape)
    boxes *= 300
    boxes = boxes.astype(np.int64)
    print("target is ", boxes)
    if boxes.any():
        for tar in boxes:
            #print("tar is ", tar)
            imgCV = cv2.rectangle(imgCV, (tar[0], tar[1]), (tar[2], tar[3]), (255, 255, 0), 3)
    cv2.imshow("image", imgCV)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
