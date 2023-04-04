import math
from __init__  import load_model,PROJECT_DIR
import os
from paddle.io import DataLoader 
from paddle.vision import transforms
from sklearn.model_selection import train_test_split 
from datapreprocess import pipline_data_mlp
from dataset import MLPDataset
from models import WordImageSliceMLPCLS
import paddle.optimizer as optim
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy,accuracy
data_list=pipline_data_mlp(dataset_dir=os.path.join(PROJECT_DIR,"mocov1","dataset"),expansion=3)
#train_data,test_data=train_test_split(data_list,test_size=0.2)
train_data=data_list
encoder_q_model,encoder_k_model=load_model("tmp/checkpoint/epoch_011_bitchth_003500_model.pdparams")
cls_model=WordImageSliceMLPCLS(encoder_model_k=encoder_k_model,encoder_model_q=encoder_k_model)

normalize = transforms.Normalize(
        mean=[0.485], std=[0.229]
    )
    # 咱们就先弄mocov1的数据增强
augmentation = [
            #transforms.RandomResizedCrop((16,48), scale=(0.2, 1.0)),
            #transforms.RandomGrayscale(p=0.2), 啥也别说了，paddle没有这个功能
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            #transforms.RandomHorizontalFlip(),训练的时候可以加上，但是不训练的加上感觉没什么用？可能起到坏处
            transforms.ToTensor(),
            normalize,
        ]
train_dataset=MLPDataset(train_data,transform=transforms.Compose(augmentation))
train_loader=DataLoader(train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        batch_sampler=None,
        drop_last=False,
    )
class MYLR(optim.lr.LRScheduler):
    """ 
    自定义的一个学习率曲线
    """

    def __init__(self,learning_rate=0.1,epochs=500, last_epoch=-1, verbose=False,cos=False,schedule=[120, 160]):
        
        self.cos=cos
        self.epochs=epochs# 总共的epoch数量
        self.schedule=schedule
        super(MYLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.cos:
            return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.epochs))
        else: 
            lr=self.base_lr
            for milestone in self.schedule:
                lr *= 0.1 if self.last_epoch >= milestone else 1.0
            return lr
lr=MYLR(learning_rate=0.001,cos=True,verbose=True)
optimizer = optim.SGD(
        learning_rate=lr,
        parameters=cls_model.parameters(),
        weight_decay=1e-4,
    )# pytorch 对应的优化器是SGD
celoss=CrossEntropyLoss()
cls_model.train()
for epoch in range(2000): 
    allloss=[]
    allacc=[]
    for i, (batch_image,batch_image_type,batch_image_import_flag) in enumerate(train_loader):
        output=cls_model(batch_image[0])
        loss=celoss(output,batch_image_type)
        acc = accuracy(output, batch_image_type.unsqueeze(1))
        #print("epoch:{}->batch:{}->loss:{}->acc:{}".format(epoch,i,float(loss),float(acc)))
        allloss.append(float(loss))
        allacc.append(float(acc))
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
    print(epoch,"acc:",sum(allacc)/(len(allacc)+0.1),"loss:",sum(allloss)/(len(allloss)+0.1))
    lr.step(epoch)

