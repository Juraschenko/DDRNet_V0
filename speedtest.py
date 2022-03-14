import time
import torch
import tensorrt as trt
from torch2trt import torch2trt
from DDRNet_23_slim import DualResNet, BasicBlock
from utils import load_checkpoint

DEVICE = "cuda"
MODEL_PATH = "DDRNET_pretrained_Imagenet_few_classes.pth.tar"


def test_fps(model, image_size):
    device = torch.device('cuda')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    model.eval()
    model.to(device)
    iterations = None
    print(model)
    input = torch.randn(1, 3, image_size[0], image_size[1]).cuda()
    model_trt = torch2trt(model, [input])
    with torch.no_grad():
        for _ in range(10):
            pred = model_trt(input)

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


if __name__ == "__main__":
    image_size = (512, 1024)
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=7, planes=32, spp_planes=128, head_planes=64,
                       augment=True)
    print(model)
    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    load_checkpoint(torch.load(MODEL_PATH), model)

    test_fps(model, image_size)
