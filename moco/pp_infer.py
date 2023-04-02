"""
1. 模型推理，加载模型
2. 随便赵一张图片
3. 切割然后做聚类，进行训练。
"""
import random
import time
from  paddle import fluid
from data.dataset import pickle_data_proc_image,WIPDataset
from moco.resnetmodels import HackResNet  
import moco.loader
import paddle
#import torch.backends.cudnn as cudnn

#import torch.distributed as dist
import paddle.distributed as dit
#import torch.multiprocessing as mp
#import paddle.

#import torch.nn as nn
import paddle.nn as nn 
import paddle.distributed.parallel
import paddle.optimizer as optim
from moco.resnetmodels import HackResNet
import moco.builder
from paddle.io import DataLoader
import paddle.vision.transforms as transforms


def load_model(model_path):
    model = moco.builder.MoCo(
        HackResNet
    )
    model.set_state_dict(paddle.load(model_path))
    print(model)
    encoder_k_model=model.encoder_k
    encoder_q_model=model.encoder_k
    return encoder_q_model,encoder_k_model 

def show_word_pice_dataset(dataset):
    wip=dataset
    from matplotlib import pyplot as plt
    offset=random.randint(30,3000)
    for x in range(16):
        plt.subplot(1,16,x+1)
        plt.imshow(wip[offset+x][0][2])
    plt.show()
def pipline01():
    """
    加载模型，运行聚类，看一下效果
    """
    normalize = transforms.Normalize(
        mean=[0.485], std=[0.229]
    )
    # 咱们就先弄mocov1的数据增强
    augmentation = [
            #transforms.RandomResizedCrop((16,48), scale=(0.2, 1.0)),
            #transforms.RandomGrayscale(p=0.2), 啥也别说了，paddle没有这个功能
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    encoder_q_model,encoder_k_model=load_model("tmp/checkpoint/epoch_011_bitchth_003500_model.pdparams")
    #model_ds=WIPDataset(data_dir="tmp/project_ocrSentences/1954-02/1954-02_05_02")# 这个是原始数据集使用
    model_ds=WIPDataset(data_dir="tmp/project_ocrSentences/1954-02/1954-02_05_02",transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))#这个是模型使用，要对数据做一些变化。
    show_word_pice_dataset(model_ds)
    train_loader = DataLoader(
        model_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        #pin_memory=True, paddle 没有过
        #sampler=None,
        drop_last=True,
    )
    for i, (images, _) in enumerate(train_loader):
        print(i)
        vec_list=encoder_k_model(images[0])
        print(len(vec_list))



if __name__=="__main__":
    pipline01()
    time.sleep(20)

        



     



