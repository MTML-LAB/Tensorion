from fastervit import create_model
import torch
import torch.nn as nn


def VIT_0(num_classes: int = 101):
    model = create_model("faster_vit_0_224", from_pretrained=True)
    model.head = nn.Linear(512, num_classes)
    return model


def test():
    model = VIT_0().to("cuda")
    print(model.head)
    c_grad = 0
    c = 0
    for p in model.parameters():
        if p.requires_grad:
            c_grad += p.numel()
        c += p.numel()

    print(f"{c=} \t {c_grad=}")


if __name__ == "__main__":
    test()
