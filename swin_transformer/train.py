import os
import torch
import torchvision.datasets as datasets
import torchvision.models
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from swin_transformer_model import Swin_Transformer_b_patch4_window7_224

def load_partial_weight(model, weight):
    weight=weight["model"]
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in weight.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

def get_imagenet(root, transform=None, target_transform=None):
    root = os.path.join(root, 'ILSVRC2012_img_val/val')
    data= datasets.ImageFolder(root=root,
                                transform=transform,
                                target_transform=target_transform)
    return data


root = 'D:\Jonior_Learn'
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
])
data = get_imagenet(root, trans)
batch_size=4
train_loader = DataLoader(data,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True,
                          )

# weight_1=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
# weight_2=torch.load("D:/Jonior_Learn/vit_b_16-c867db91.pth")
# model=torchvision.models.vit_b_16(weights=weight_1)

model =Swin_Transformer_b_patch4_window7_224()
weight=torch.load("D:\Jonior_Learn\swin_base_patch4_window7_224.pth")
model=load_partial_weight(model, weight)
dic=model.state_dict()
model=model.cuda()
print(model)

cnt=0
times=0
matrix=torch.zeros(1000,1000).cuda()
for img, label in train_loader:
    img=img.cuda()
    label=label.cuda()
    out=model(img)
    matrix[label,out.argmax(1)]=matrix[label,out.argmax(1)]+1
    cnt = cnt + (label == out.argmax(1)).sum()
    times = times + 1
    if times==2:
        break

matrix=matrix.cpu()
from scipy.io import savemat
import numpy as np
result = np.array(matrix)
savemat("matlab_matrix.mat", {'result1':result})
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.5)
plt.rc('font',family='Times New Roman',size=12)
sns.heatmap(matrix,center=matrix[0,0])
plt.show()

acc=cnt/(times*batch_size)
print(acc)

