# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import glob
import datetime

from utils.trainDataset import trainDataset
from utils.mobile_model import create_mobilenetv1_ssd
from utils.trainTransform import TrainAugmentation
from layers.functions.detection import *
from layers.functions.prior_box import *
from layers.modules.multibox_loss import MultiBoxLoss


from config import mb1_cfg, MEANS, SIZE

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class TrainEngine():
    def __init__(self, dataloader, network, device, epoch=300):

        self.network = network
        self.dataloader = dataloader

        self.epoch = epoch
        self.device = device

        self.creterion = MultiBoxLoss(mb1_cfg["num_classes"], 0.5, True, 0, True, 3, 0.5,
                                      False, use_gpu=True)
        self.writer = SummaryWriter(log_dir="./logs")

    def __call__(self):
        print("network is ", self.network)
        self.network = self.network.to(self.device)
        # self.network.apply(self.init_weights)

        #optimizer = optim.Adam(self.network.parameters(), lr = 0.0005)
        optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.1,
                              weight_decay=0.0001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[240, 260, 280], gamma=0.9)

        for epo in tqdm(range(self.epoch)):
            for i, (img, anno) in enumerate(self.dataloader):
                #print(img.size())
                img = img.to(self.device)
                anno = [ann.to(self.device) for ann in anno]
                torch.autograd.set_detect_anomaly(True)
                self.network.train()
                optimizer.zero_grad()
                result = self.network(img)
                #print("result is \n", result)
                #print("anno is ", result)
                loss_loc, loss_conf = self.creterion(result, anno)
                loss = loss_loc + loss_conf
                print("loss", loss, "lr =", scheduler.get_lr()[0])

                self.writer.add_scalar("loss", loss, epo+i)
                #self.writer.add_scalar("y", loss, epo+i)

                if not torch.isnan(loss): # lossがnanの場合, 学習が壊れるので避ける
                    if i != 0:
                        loss.backward(retain_graph=True)

                optimizer.step() # 最適化(ウェイトの更新)
            scheduler.step()

            if epo % 1 == 0:
                torch.save(self.network.state_dict(), "./weight/state_{}.pth".format(epo)) # ウェイトの保存
        self.writer.close()
        torch.save(self.network.state_dict(), "./weight/ssd300_210523_{}.pth".format(datetime.datetime.now()))


dataset_root = "data/201104_anno_split8-singleclass"
base_weightPath = "weight/base_weight/models/mobilenet_v1_with_relu_69_5.pth"

cfg = mb1_cfg
mean_cfg = MEANS
size_cfg = SIZE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = build_ssd("train", cfg["min_dim"], cfg["num_classes"])
net = create_mobilenetv1_ssd(2)

# from utils.ssd_model import SSD
# net2 = SSD("test", data_cfg)

base_weight = torch.load(base_weightPath)
net.base_net.load_state_dict(base_weight)
# net.extras.apply(weight_init)
# net.loc.apply(weight_init)
# net.conf.apply(weight_init)
net.base_net.apply(weight_init)
net.extras.apply(weight_init)
net.classification_headers.apply(weight_init)
net.regression_headers.apply(weight_init)

transform = TrainAugmentation()
Dataset = trainDataset(dataset_root, transform=transform)
train_dataloader = torch.utils.data.DataLoader(
    Dataset,
    batch_size = 8,
    shuffle=True,
    collate_fn = detection_collate
)
trainEngine = TrainEngine(train_dataloader, net, device)

trainEngine()


