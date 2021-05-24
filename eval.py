import numpy as np
import torch
import torch.nn as nn
import cv2

from config import data_cfg, MEANS, SIZE
from utils.model import build_ssd
from utils.evalDataset import EvalDataset
from utils.evalTransform import EvalAugmentation

def check_image(img):
    imgCV = img
    #imgCV = img.cpu().detach().numpy().copy()
    #print("img is ", imgCV.shape[0])
    #if 3 == imgCV.shape[0]:
    #    imgCV = imgCV.transpose(1, 2, 0).astype(np.uint8)
    #if 4 == imgCV.shape[0]:
    #imgCV = np.concatenate([imgCV[0], imgCV[1], imgCV[2], imgCV[3]], 2)
    #imgCV = imgCV.transpose(1, 2, 0).astype(np.uint8)
    #print("img is ", imgCV.shape)

    cv2.imshow("imageeval", imgCV)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        #print("sample[0] is ", sample[0].shape)
        try:
            if sample[1].any():
                targets.append(torch.FloatTensor(sample[1]))
        except (TypeError, AttributeError):
            #print("target is None")
            continue
        #print("target is ", targets)

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets

class EvalEngine():
    def __init__(self, dataloader, net, device, weight):
        self.net = net
        self.dataloader = dataloader
        self.device = device
        self.net.eval()
        self.net.load_state_dict(torch.load(weight))

    def __call__(self):
        self.network = self.net.to(self.device)

        for i, (img, anno) in enumerate(self.dataloader):
            img = img.squeeze(0)
            #check_image(img[0])
            #print("img is ", img)
            img = img.to(self.device)
            img = img.float()
            #print("img shape is ", img.shape, "anno is ", anno)
            detections = self.net(img)
            #print("detections is ", detections, "anno is ", anno)
            print("anno is ", anno)
            self._extract_thresholdBox(img, detections, threshold=0.5)

    def _extract_thresholdBox(self, img_ori, detections, threshold=0.5):
        width = 300
        height = 300
        predict_bbox = []
        predict_label_index = []
        predict_scores = []
        detections = detections.cpu().detach().numpy().copy()
        print("detections is ", detections.shape)
        #print("detections is \n", detections[:, 0:, :, 0])
        for parts, detectbox in enumerate(detections):
            find_index = np.where(detectbox[0:, :, 0] >= threshold)
            #print("np.where is ", detections[:, 0:, :, 0] >= threshold)
            #print("find_index is ", find_index)
            detectbox = detectbox[find_index]
            #print("detections is ", detectbox.shape)

            #print("detections is ", detections)
            # select part of image because position of image is different between each of batch
            for part_detection in detectbox:
                print("part_detection is ", part_detection)
                score = part_detection[0]
                bbox = part_detection[1:] * [width, height, width, height]
                bbox = bbox + [width*parts, 0, width*parts, 0]
                bbox = bbox.astype(np.int16)
                label_index = 1  # HACK this detection only have "human" label.
                predict_bbox.append(bbox)

        imgCV = img_ori.cpu().detach().numpy().copy()
        imgCV = np.concatenate([imgCV[0], imgCV[1], imgCV[2], imgCV[3]], 2)
        imgCV = imgCV.transpose(1, 2, 0)
        imgCV += [104, 117, 123]
        imgCV = imgCV.astype(np.uint8)

        for bo in predict_bbox:
            imgCV = cv2.UMat(imgCV)
            imgCV = cv2.rectangle(imgCV, (bo[0], bo[1]), (bo[2], bo[3]), (255, 255, 0), 5)

        check_image(imgCV)

dataset_root = "data/20.09.30_annotation"
weight = "weight/ssd300_210523.pth"

cfg = data_cfg
mean_cfg = MEANS
size_cfg = SIZE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = build_ssd("test", cfg["min_dim"], cfg["num_classes"])
from utils.ssd_model import SSD
net2 = SSD("test", data_cfg)
print("net is ", net)
print("other net is ", net2)
transform = EvalAugmentation()
Dataset = EvalDataset(dataset_root, transform=transform)
eval_dataloader = torch.utils.data.DataLoader(
    Dataset,
    batch_size = 1, # HACK: You Must fix to 1
    collate_fn = detection_collate
)
evalEngine = EvalEngine(eval_dataloader, net, device, weight)

evalEngine()










