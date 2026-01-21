from fastervit import create_model
import torch
import torch.nn as nn

def VIT_0(num_classes: int=101):
    model = create_model('faster_vit_1_224')
    model.head = nn.Linear(640, num_classes)
    return model

def test():
    model = VIT_0().to('cuda')
    print(model.head)
    x = torch.randn((1, 3, 224, 224), device='cuda')
    y = model(x)
    print(y.shape)
    # print(list(model.children()))
    c = 0
    for p in model.parameters():
        c += p.numel()

    print(c)

if __name__ == "__main__":
    test()
