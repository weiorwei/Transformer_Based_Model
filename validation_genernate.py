import scipy.io as scio  # import scipy; scipy.io 报错 玄学！！
import os
import shutil
import numpy as np
import torch.cuda


def move_valimg(val_dir='D:\Jonior_Learn\ILSVRC2013_DET_val', devkit_dir='D:\Jonior_Learn\ILSVRC2013_devkit'):
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
    synset = scio.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))

    label_map = open("E:/DL/transformer/label_map.txt").read()
    label_map = eval(label_map)
    print(type(label_map))
    print(np.array(list(label_map.values()))[:, 0])
    label_map = np.array(list(label_map.values()))[:, 0]

    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(val_dir))

    all_labels = []
    for filename in filenames:
        print(filename)
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))
        print(np.where(label_map == WIND))
        img_cls = np.where(label_map == WIND)[0]
        all_labels.append("%s\t%d" % (filename, img_cls))
        # break

    print(all_labels)
    with open("E:/DL/transformer/val_map.txt", "w") as f:
        for i in range(len(all_labels)):
            f.write(str(all_labels[i]) + '\n')

torch.cuda.is_available()
if __name__ == "__main__":
    move_valimg()


