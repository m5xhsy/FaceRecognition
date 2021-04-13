import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def get_transform(mode):
    if mode == "train":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(160),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform
    else:
        transform = transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

def get_all_dataloader(data_dir, batch_size, num_workers=1):
    transform = get_transform()

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    verify_dir = os.path.join(data_dir, "verify")

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    verify_dataset = ImageFolder(root=verify_dir, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    verify_loader = DataLoader(dataset=verify_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)

    return train_loader, verify_loader, test_loader


def get_dataloader(data_dir, mode, batch_size, num_workers=1):
    transform = get_transform(mode)
    path = os.path.join(data_dir, mode)

    dataloader = DataLoader(
        dataset=ImageFolder(root=path, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return dataloader


class Err(Exception):
    pass

def get_num_classes(data_dir):
    num = 0
    for item in os.listdir(data_dir):
        path = os.path.join(data_dir,item)
        if num != 0 and num != len(os.listdir(path)):
            raise Err("数据集不匹配")
        else:
            num = len(os.listdir(path))
    return num


