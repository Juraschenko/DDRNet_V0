import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import albumentations as A
import torchmetrics
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from utils import load_checkpoint, save_checkpoint, dice_coefficient, iou_coefficient, dice_loss, intersection_over_union
import torch.optim as optim
from dataset import getloader, decode_segmap, encode_segmap
import dataset
from DDRNet_23_slim import get_seg_model
from DDRNet_23_slim import DualResNet, BasicBlock

writer = SummaryWriter("runs/DDRNet_test")

BatchNorm2d = nn.BatchNorm2d

bn_mom = 0.1
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
NUM_EPOCHS = 250
FEW_CLASSES = True


NUM_WORKER = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024
PIN_MEMORY = True
LOAD_MODEL = True
INSPECT_DATA = False
# MODEL_PATH = "DDR_model_custom_few_classes.pth.tar" --> 63er IOU ihne image net
MODEL_PATH = "DDRNET_pretrained_Imagenet_few_classes.pth.tar"


class Model:
    def __init__(self):
        self.alpha = 0.4
        self.step_global = 0
        self.ignore_index = 255
        if FEW_CLASSES:
            self.class_names = ["unlabeled", "ego vehicle",'road', "pole", "vegetation", "terrain", "sky"]
            self.void_classes = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, -1]
            self.valid_classes = [self.ignore_index, 1, 7, 17, 21,  22, 23]
            self.colors = [(0, 0, 0), (0, 0, 0), (128, 64, 128), (153, 153, 153), (107, 142, 35) , (152, 251, 152), (70, 130, 180) ]
        else:

            self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 29, 30, -1]
            self.valid_classes = [self.ignore_index, 7, 8, 11, 17, 21, 22, 23, 24, 25, 26, 27, 28, ]
            self.class_names = ["unlabeled", 'road', 'sidewalk', 'building', "pole", "vegetation", "terrain", "sky",
                                "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                                "bicycle"]
            self.colors = [(0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (153, 153, 153),
                           (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                           (255, 0, 0),
                           (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))
        self.n_classes = len(self.valid_classes)
            # print(class_map)
            # print(n_classes)

        self.label_colors = dict(zip(range(self.n_classes), self.colors))
        self.train_transform = A.Compose(
        [
            A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.Crop(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.486), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.val_transform = A.Compose(
        [
            A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
            A.Normalize(mean=(0.485, 0.456, 0.486), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.train_loader, self.val_loader = getloader(
            train_transform=self.train_transform,
            batch_size=BATCH_SIZE,
            val_transform=self.val_transform,
            num_workers=NUM_WORKER,
            pin_memory=PIN_MEMORY,
        )

    def process(self, image, segment):
        out = image
        segment = dataset.encode_segmap(segment, self.void_classes, self.ignore_index, self.valid_classes)
        return out, segment

    def train(self, data, model, optimizer, loss_fn, scaler):
        loop = tqdm(data)
        running_loss = 0
        if INSPECT_DATA:
            for batch_idx, (data, targets) in enumerate(loop):
                for i in range(len(data)):
                    image = data[i, ...][None, ...]
                    print(targets.shape)
                    segment = targets[i, ...][None, ...]
                    print(segment.shape)
                    segment = torch.squeeze(segment)
                    print(segment.shape)
                    image, segment = self.process(image, segment)
                    # print(image.shape, segment.shape)
                    resl = decode_segmap(segment.clone(), self.label_colors, self.n_classes)
                    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
                    ax[0].imshow(segment, cmap="gray")
                    ax[1].imshow(resl)
                    plt.show()

        for batch_idx, (data, targets) in enumerate(loop):
            image, segment = self.process(data, targets)
            # print(image.shape, segment.shape)
            image = image.to(device=DEVICE)
            segment = segment.long().to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                prediction, out_aux = model(image)
                loss_n = loss_fn(prediction, segment)
                loss_aux = loss_fn(out_aux, segment)
                loss = loss_n + loss_aux * self.alpha

            # writer.add_scalar("Loss/train", loss, )
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            # update tqdm loop
            loop.set_postfix(loss=loss.item())
        writer.add_scalar("training_loss", running_loss / batch_idx, global_step=self.step_global)
        # writer.close()

    def run(self):
        model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.n_classes, planes=32, spp_planes=128, head_planes=64, augment=True).to(DEVICE)
        # model = get_seg_model(self.n_classes).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()

        # writer.add_graph(model, (BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH))
        # writer.close()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005, momentum=0.9)
        # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        metrics = torchmetrics.IoU(self.n_classes)

        if LOAD_MODEL:
            load_checkpoint(torch.load(MODEL_PATH), model)

        # check_accuracy(val_loader, model, device=DEVICE)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(NUM_EPOCHS):
            print("Epoche {} ------------------------".format(epoch))
            self.step_global = epoch
            self.train(self.train_loader, model, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=MODEL_PATH)

            model.eval()
            total_iou = 0
            if epoch % 5 == 0:
                with torch.no_grad():
                    for idx, (x, y) in enumerate(self.val_loader):
                        mean_iou = []
                        image, segment = self.process(x, y)

                        image = image.to(DEVICE)
                        segment = segment.long().to(DEVICE)

                        preds = model(image)[0]
                        preds = torch.argmax(preds, 1)
                        for i in range(len(x)):
                            image = x[i, ...][None, ...]
                            # print(x.shape)
                            segment = y[i, ...][None, ...]
                            # print(segment.shape)
                            segment = torch.squeeze(segment)
                            # print(segment.shape)
                            pred = preds[i, ...][None, ...]
                            # print(preds.shape)
                            pred = torch.squeeze(pred)

                            if idx == 10 and epoch == 0:
                                resl = decode_segmap(pred.cpu(), self.label_colors, self.n_classes)
                                decoded_gt = decode_segmap(segment.clone(), self.label_colors, self.n_classes)
                                fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
                                ax[0].imshow(decoded_gt)
                                ax[1].imshow(resl)
                                plt.show()
                            iou = intersection_over_union(pred, segment, self.n_classes)

                            if mean_iou == []:
                                mean_iou = iou

                            else:
                                mean_iou += iou

                            new_iou = [x for x in mean_iou.tolist() if math.isnan(x) == False]
                            new_iou = np.array(new_iou)
                            batch_iou = new_iou.mean()/(i + 1)
                            # print("Mean IOU:" + str(new_iou / (i + 1)))
                            # print("IOU over all classes: " + str(batch_iou))
                            mean_iou.tolist()

                        total_iou += batch_iou
                        mean_total_iou = total_iou/(idx + 1)
                    #writer.add_scalar("Validation IoU", mean_total_iou, global_step=self.step_global)
                    print("Mean IOU over all batches: " + str(mean_total_iou))
                #self.step_global += 1
            model.train()

    def eval_model(self):
        """
        Function to evaluate model on test dataset
        :return:
        """
        model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.n_classes, planes=32, spp_planes=128, head_planes=64, augment=True).to(DEVICE)
        print(model)
        # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        load_checkpoint(torch.load(MODEL_PATH), model)

        model.eval()
        model.to(DEVICE)
        print(model)
        total_iou = 0

        with torch.no_grad():
            for idx, (x, y) in enumerate(self.val_loader):
                mean_iou = []
                image, segment = self.process(x, y)
                size = segment.size()
                image = image.to(DEVICE)
                segment = segment.long().to(DEVICE)

                preds = model(image)[0]

                # preds = F.upsample(input=preds, size=(
                #     size[-2], size[-1]), mode='bilinear')
                preds = torch.argmax(preds, dim=1)
                for i in range(len(x)):
                    image = x[i, ...][None, ...]
                    # print(x.shape)
                    segment = y[i, ...][None, ...]
                    # print(segment.shape)
                    segment = torch.squeeze(segment)
                    # print(segment.shape)
                    pred = preds[i, ...][None, ...]
                    # print(preds.shape)
                    pred = torch.squeeze(pred)

                    resl = decode_segmap(pred.cpu(), self.label_colors, self.n_classes)
                    decoded_gt = decode_segmap(segment.clone(), self.label_colors, self.n_classes)
                    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
                    ax[0].imshow(decoded_gt)
                    ax[1].imshow(resl)
                    plt.show()
                    iou = intersection_over_union(pred, segment, self.n_classes)

                    # print(iou)
                    if mean_iou == []:
                        mean_iou = iou

                    else:
                        mean_iou += iou

                    new_iou = [x for x in mean_iou.tolist() if math.isnan(x) == False]
                    new_iou = np.array(new_iou)
                    batch_iou = new_iou.mean() / (i + 1)
                    # print("Mean IOU:" + str(new_iou / (i + 1)))
                    # print("IOU over all classes: " + str(batch_iou))

                total_iou += batch_iou
                mean_total_iou = total_iou / (idx + 1)
            print("Mean IOU over all batches: " + str(mean_total_iou))


if __name__ == '__main__':
    experiment = Model()
   #experiment.eval_model()
    experiment.run()
