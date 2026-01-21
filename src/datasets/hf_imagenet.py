import torchvision.transforms as transforms
from torch.utils.data import Dataset


class HF_imagenet(Dataset):
    def __init__(self, dataset, transform):
        self.ds = dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = self.transform(item["image"])

        return img, item["label"]


def _load_imaginet(download=True, data_path=None):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            # check_shape(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            # check_shape(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ds = load_from_disk("ILSVRC/imagenet-1k")

    train_ds = HF_imagenet(ds["train"], transform=transform_train)
    test_ds = HF_imagenet(ds["validation"], transform=transform_test)

    return train_ds, test_ds


def load_from_disk(ds_name):
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("datasets is required for ImageNet loading") from exc

    path = f"/data/datasets/{ds_name.replace('/', '_')}"
    dataset = datasets.load_from_disk(path)
    print(dataset)
    return dataset
