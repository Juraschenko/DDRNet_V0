import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Any, Tuple
from torchvision.datasets import Cityscapes
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

#ROOT = r"D:\Hochschule\Master\3_Semester\Projektarbeit\Testtrack_Dataset"
ROOT = r"C:\Users\User\PycharmProjects\PytorchTest\data"


def class_mapping(mask, mapping):
    for key in mapping:
        mask[mask == key] = mapping[key]
    return mask


def encode_segmap(mask, void_classes, ignore_index, valid_classes):
    """function to delete unwanted classes and rectify wanted classes"""

    class_map = dict(zip(valid_classes, range(len(valid_classes))))

    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


def decode_segmap(temp, label_colors, n_classes):
    """convert grayscale to color"""
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colors[l][0]
        g[temp == l] = label_colors[l][1]
        b[temp == l] = label_colors[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


class MyCityScapes(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert("RGB")

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed = self.transforms(image=np.array(image), mask=np.array(target))

            return transformed["image"], transformed["mask"]
        else:
            return image, target


def getloader(
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,):

    train_ds = MyCityScapes(ROOT, split="train", mode="fine", target_type="semantic", transforms=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True)

    val_ds = MyCityScapes(ROOT, split="val", mode="fine", target_type="semantic", transforms=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                            shuffle=False)

    return train_loader, val_loader

### Test Area ###


def main():
    ignore_index = 255
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    class_names = ["unlabeled", 'road', 'sidewalk', 'building', "wall", "fence", "pole", "traffic_light",
                   "traffic_sign",
                   "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                   "bicycle"]

    class_map = dict(zip(valid_classes, range(len(valid_classes))))
    n_classes = len(valid_classes)
    # print(class_map)
    # print(n_classes)
    colors = [(0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
              (153, 153, 153),
              (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
              (255, 0, 0),
              (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

    label_colors = dict(zip(range(n_classes), colors))
    transform = A.Compose(
        [
            A.Resize(256, 512),
            A.HorizontalFlip(),
            A.Normalize(mean=(0.485, 0.456, 0.486), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    mapping = {1: ignore_index, # vehicle
               2: 17, # vegetation
               3: 7, # road
               4: 23, # 4 --> sky
               5: 21, # pole
               6: 22,
               }
    # sky -> 0, road -> 1, vegetation -> 2, terrain 3, ego veh 4, pole -> 5

    """Test to load data"""
    dataset = MyCityScapes(ROOT, split="train", mode="fine", target_type="semantic", transforms=transform)
    for i in range(0, 5):
        img, seg = dataset[i]
        print(dataset.images_dir)
        # print(img.shape, seg.shape)

        #seg = class_mapping(seg, mapping)

        # fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
        # ax[0].imshow(img.permute(1, 2, 0))
        # ax[1].imshow(seg, cmap="gray")
        # plt.show()

        # print(torch.unique(seg))

        res = encode_segmap(seg.clone(), void_classes, ignore_index, valid_classes)
    #print(res.shape)
    # print(torch.unique(res))

        resl = decode_segmap(res.clone(), label_colors, n_classes)

        """display properly colored images"""
        fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
        ax[0].imshow(res, cmap="gray")
        ax[1].imshow(resl)

        plt.show()


if __name__ == "__main__":
    main()
