import torch
import numpy as np
import torchvision
from dataset import encode_segmap, decode_segmap
import time


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def eval_model(model, test_loader):
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            img, seg = batch
            output = model(img.cuda())
    print(img.shape, seg.shape, output.shape)
    sample = 6

    outputx = output.detach().cpu()[sample]
    # encoded_mask =


def dice_loss(y_pred, y_true):
    """

    :param y_pred:
    :param y_true:
    :return:
    """
    smooth = 1
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    intersection = (y_pred_flat * y_true_flat).sum()

    return 1 - ((2. * intersection + smooth) /
                (y_pred_flat.sum() + y_true_flat.sum() + smooth))


def dice_coefficient(y_true, y_pred):
    num = 2 * torch.sum(y_true * y_pred)
    den = torch.sum(y_true + y_pred)
    return num / (den + 1e-7)


def iou_coefficient(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred))
    union = torch.sum(y_true , [1,2,3]) + torch.sum(y_pred, [1,2,3]) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    return iou


def intersection_over_union(y_pred, y_true, n_classes):
    ious = []
    for cls in range(1, n_classes):
        pred_inds = y_pred == cls
        target_inds = y_true == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()
        union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)


def test_fps(model, image_size):
    device = torch.device('cuda')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    model.eval()
    model.to(device)
    iterations = None
    print(model)
    input = torch.randn(1, 3, image_size[0], image_size[1]).cuda()
    with torch.no_grad():
        for _ in range(10):
            pred = model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
