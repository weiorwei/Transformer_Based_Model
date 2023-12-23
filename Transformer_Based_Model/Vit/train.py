import os
import torch
import torchvision.datasets as datasets
import torchvision.models
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from vit_model import vit_base_patch16_224_in21k

def load_partial_weight(model, weight):
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in weight.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

def get_imagenet(root, transform=None, target_transform=None):
    root = os.path.join(root, 'ILSVRC2012_img_val')
    data= datasets.ImageFolder(root=root,
                                transform=transform,
                                target_transform=target_transform)
    return data


root = 'D:\Jonior_Learn'
trans = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
data = get_imagenet(root, trans)
batch_size=12
train_loader = DataLoader(data,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True,
                          )

weight_1=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
weight_2=torch.load("D:/Jonior_Learn/vit_b_16-c867db91.pth")
model=torchvision.models.vit_b_16(weights=weight_1)

# weight=torch.load("D:/Jonior_Learn/vit_jx_mixer_b16_224-76587d61.pth")
# model=load_partial_weight(model, weight)
model=model.cuda()
print(model)

cnt=0
times=0
for img, label in train_loader:
    img=img.cuda()
    label=label.cuda()
    out=model(img)
    cnt = cnt + (label == out.argmax(1)).sum()
    times = times + 1

acc=cnt/(times*batch_size)
print(acc)