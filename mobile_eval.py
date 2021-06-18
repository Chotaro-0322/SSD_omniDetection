import numpy as np
import torch
import torch.nn as nn
import cv2


from config import data_cfg, MEANS, SIZE
# from utils.model import build_ssd
from utils.evalDataset import EvalDataset
# from utils.evalTransform import EvalAugmentation

from utils.mobile_model import create_mobilenetv1_ssd
from utils.evalTransform import EvalAugmentation, AfterAugmentation
# from layers.functions.detection import *
from utils.prior_box import *
from utils.predictor import Predictor

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

        self.net.to(torch.device("cuda:0"))

        net_weights = torch.load("/home/itolab-chotaro/HDD/Python/210518_SSD_graduate/weight/210618_ssd_mb1_ver2/mb1_100.pth",
                                map_location={'cuda:0': 'cpu'})

        self.net.load_state_dict(net_weights)

        #print("into Object detection!!!!")

        self.after_transform = AfterAugmentation()

        self.Predictor = Predictor(candidate_size=200)

    def __call__(self):
        for i, (img, anno) in enumerate(self.dataloader):
            boxes_list = []
            score_list = []
            img = img.squeeze(0)
            # img, _, _ = self.transform(img, "", "")
            # print("img is", img.size())
            img = img.to(torch.device("cuda:0"))
            # print("img is ", img)
            scores, boxes = self.net(img)
            # print("boxes is ", boxes.size())
            img = img.to(torch.device("cpu"))
            img_ori, boxes, _ = self.after_transform(img, boxes, "")
            # img, _, _ = self.after_transform(img, "", "")
            # print("img is ", img.shape)
            # boxes, labels, score = self.Predictor.predict(score, boxes, 10, 0.4)
            for box, score in zip(boxes, scores):
                # print("box is ", box.shape)
                box, labels, score = self.Predictor.predict(score, box, 100, 0.4)
                #print("box is ", box)
                boxes_list.append(box)
                score_list.append(score)
            # print("boxes is ", boxes)
            # print("labels is ", labels)

            img_ori = np.uint8(img_ori)
            boxes = boxes.to('cpu').detach().numpy().copy()

            # print("img_ori is ", img_ori)
            img_ori = cv2.resize(img_ori, (1200, 300)) #よくわからないけど, これがないとエラー
            img_ori = cv2.line(img_ori, (300, 0), (300, 300), (0, 255, 0), 2)
            img_ori = cv2.line(img_ori, (900, 0), (900, 300), (0, 255, 0), 2)
            height, width, _ = img_ori.shape
            print(type(img_ori))
            # print("img is ", img)
            # print("box is ", prediction_box)
            only_front = False

            if only_front == True:
                for i, boxes in enumerate(boxes_list):
                    if (boxes is not None) and (i == 0):
                        for box in boxes:
                            # print("box is ", box)
                            # print("box is ", np.int(box[1] + 1 * width/4))
                            img_ori = cv2.rectangle(img_ori,
                                                (np.int(box[0] + 1 * width/4), np.int(box[1])),
                                                (np.int(box[2] + 1 * width/4), np.int(box[3])),
                                                (255, 0, 0), 5
                                                )
                    elif (boxes is not None) and i == 1:
                        for box in boxes:
                            img_ori = cv2.rectangle(img_ori, (np.int(box[0] + 2 * width/4), np.int(box[1])), (np.int(box[2] + 2 * width/4), np.int(box[3])), (255, 0, 0), 5)
            else:
                print ("only_front is False")
                for i, boxes in enumerate(boxes_list):
                    if (boxes is not None):
                        for box in boxes:
                            img_ori = cv2.rectangle(img_ori,
                                                (np.int(box[0] + i * width/4), np.int(box[1])),
                                                (np.int(box[2] + i * width/4), np.int(box[3])),
                                                (255, 0, 0), 5
                                                )
            self.check_image(img_ori)

    def check_image(self, imgCV):
        cv2.imshow("image", imgCV)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def _extract_thresholdBox(self, img_ori, detections, threshold=0.5):
        width = 300
        height = 300
        predict_bbox = []
        predict_label_index = []
        predict_scores = []
        detections = detections.cpu().detach().numpy().copy()
        # print("detections is ", detections.shape)
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
                # print("part_detection is ", part_detection)
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
# net = build_ssd("test", cfg["min_dim"], cfg["num_classes"])
# from utils.ssd_model import SSD
# net2 = SSD("test", data_cfg)
# print("net is ", net)
# print("other net is ", net2)
transform = EvalAugmentation()
net = create_mobilenetv1_ssd(2, is_test=True)
net.eval()
Dataset = EvalDataset(dataset_root, transform=transform)
eval_dataloader = torch.utils.data.DataLoader(
    Dataset,
    batch_size = 1, # HACK: You Must fix to 1
    collate_fn = detection_collate
)
evalEngine = EvalEngine(eval_dataloader, net, device, weight)

evalEngine()










