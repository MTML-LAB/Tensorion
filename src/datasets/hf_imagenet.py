from typing import Optional

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

    @staticmethod
    def _load_imagenet_dataset(
        download: bool,
        data_path: Optional[str],
        hf_name: str,
    ):
        try:
            import datasets
        except ImportError as exc:
            raise RuntimeError("datasets is required for ImageNet loading") from exc

        if download:
            kwargs = {}
            if data_path:
                kwargs["cache_dir"] = data_path
            dataset = datasets.load_dataset(hf_name, **kwargs)
        else:
            if not data_path:
                raise ValueError("data_path must be set when download is False")
            dataset = datasets.load_from_disk(data_path)

        print(dataset)
        return dataset


def _load_imagenet(
    download: bool = True,
    data_path: Optional[str] = None,
    hf_name: Optional[str] = "imagenet-1k",
):
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

    if hf_name is None:
        raise ValueError("The hf_name is required for ImageNet loading")

    ds = HF_imagenet._load_imagenet_dataset(
        download=download, data_path=data_path, hf_name=hf_name
    )

    train_ds = HF_imagenet(ds["train"], transform=transform_train)
    test_ds = HF_imagenet(ds["validation"], transform=transform_test)

    return train_ds, test_ds
