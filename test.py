from scipy import io
import os
import shutil
import torch

def move_valimg(val_dir='D:\Jonior_Learn\ILSVRC2012_img_val', devkit_dir='D:\Jonior_Learn\ILSVRC2012_devkit_t12\ILSVRC2012_devkit_t12'):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    # load synset, val ground truth and val images list
    synset = io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))

    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))


from numpy import load
from Vit.vit_model import vit_base_patch16_224_in21k
data = load('D:\Jonior_Learn\imagenet21k+imagenet2012_ViT-B_16-224.npz')
lst = data.files
weight=torch.load("D:\Jonior_Learn\swin_base_patch4_window7_224.pth")
for item in lst:
    print(item)
model=vit_base_patch16_224_in21k(num_classes=1000, prelogit=False)
model_dict = model.state_dict()
print()

