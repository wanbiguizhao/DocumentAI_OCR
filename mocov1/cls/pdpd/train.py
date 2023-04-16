import argparse
import random
import warnings
from __init__  import load_model,PROJECT_DIR
import os
import paddle
from paddle import nn
from paddle.io import DataLoader 
from paddle.vision import transforms
from datapreprocess import pipline_data_mlp
from dataset import MLPDataset
from models import WordImageSliceMLPCLS
import paddle.optimizer as optim
from lr import MYLR
from paddle.nn import CrossEntropyLoss
from paddle.metric import accuracy
from visualdl import LogWriter


parser = argparse.ArgumentParser(description="训练一个神经网络用于预测wordslice是汉字的一部分还是两个字之间的部分")
parser.add_argument("--data",type=str,default="mocov1/dataset", metavar="DIR", help="path to dataset,指向按行切割的图片的文件夹目录")
parser.add_argument(
    "--expansion",
    default=3,
    type=int,
    metavar="EXPAN",
    help="对于连着的图片的数据，复制倍数",
)
parser.add_argument(
    "--test-size",
    default=0.2,
    type=float,
    metavar="TS",
    help="test size of all data ",
    #dest="lr",
)
parser.add_argument("--moco_model",type=str,default="tmp/checkpoint/epoch_002_bitchth_013000_model.pdparams", metavar="Backnone", help="对比模型的存储目录")
parser.add_argument("--freeze_flag", action="store_true", help="训练模型是否冻结backbone")# paddle的学习率使用策略和pytorch不一样
parser.add_argument("--checkpoint", type=str,default="tmp/nobackbone", help="训练模型的保存位置")# paddle的学习率使用策略和pytorch不一样
parser.add_argument("--checkpoint_steps", type=int,default=15, help="每过多少轮保存一下模型")# paddle的学习率使用策略和pytorch不一样

parser.add_argument("--logdir", type=str,default="tmp/moco_cls", help="训练日志存储路径")# paddle的学习率使用策略和pytorch不一样

parser.add_argument(
    "-j",
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="加载dataset的work",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="总共训练的轮数" 
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts，主要是方便计算学习率)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help=" 这个意思莫非是，多卡GPU的情况，256被多卡平均使用，想多了，用不到多卡",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[20, 50,100],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")# paddle的学习率使用策略和pytorch不一样
parser.add_argument("--cpu", action="store_true", help="使用cpu训练")# paddle的学习率使用策略和pytorch不一样

# train_data,test_data=pipline_data_mlp(dataset_dir=os.path.join(PROJECT_DIR,"mocov1","dataset"),expansion=3)
# #train_data,test_data=train_test_split(data_list,test_size=0.2)
# encoder_q_model,encoder_k_model=load_model("tmp/checkpoint/epoch_002_bitchth_013000_model.pdparams")
# cls_model=WordImageSliceMLPCLS(encoder_model_k=encoder_k_model,encoder_model_q=encoder_q_model)

# normalize = transforms.Normalize(
#         mean=[0.485], std=[0.229]
#     )
#     # 咱们就先弄mocov1的数据增强
# augmentation = [
#             #transforms.RandomResizedCrop((16,48), scale=(0.2, 1.0)),
#             #transforms.RandomGrayscale(p=0.2), 啥也别说了，paddle没有这个功能
#             #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
#             #transforms.RandomHorizontalFlip(),训练的时候可以加上，但是不训练的加上感觉没什么用？可能起到坏处
#             transforms.ToTensor(),
#             normalize,
#         ]
# train_dataset=MLPDataset(train_data,transform=transforms.Compose(augmentation))
# train_loader=DataLoader(train_dataset,
#         batch_size=32,
#         shuffle=True,
#         num_workers=2,
#         batch_sampler=None,
#         drop_last=False,
#     )
# test_dataset=MLPDataset(test_data,transform=transforms.Compose(augmentation))
# test_loader=DataLoader(test_dataset,
#         batch_size=512,
#         shuffle=True,
#         num_workers=2,
#         batch_sampler=None,
#         drop_last=False,
#     )
# lr=MYLR(learning_rate=0.001,cos=True,verbose=True)
# optimizer = optim.SGD(
#         learning_rate=lr,
#         parameters=cls_model.parameters(),
#         weight_decay=1e-4,
#     )# pytorch 对应的优化器是SGD

# celoss=CrossEntropyLoss()

# for epoch in range(2000): 
#     allloss=[]
#     allacc=[]
#     cls_model.train()
#     for i, (batch_image,batch_image_type,batch_image_import_flag) in enumerate(train_loader):
#         output=cls_model(batch_image[0])
#         loss=celoss(output,batch_image_type)
#         acc = accuracy(output, batch_image_type.unsqueeze(1))
#         #print("epoch:{}->batch:{}->loss:{}->acc:{}".format(epoch,i,float(loss),float(acc)))
#         allloss.append(float(loss))
#         allacc.append(float(acc))
#         optimizer.clear_grad()
#         loss.backward()
#         optimizer.step()
#     print(epoch,"acc:",sum(allacc)/(len(allacc)+0.1),"loss:",sum(allloss)/(len(allloss)+0.1))
#     cls_model.eval()
#     with paddle.no_grad():
#         for j ,(batch_image,batch_image_type,batch_image_import_flag) in enumerate(test_loader):
#             output=cls_model(batch_image[0])
#             loss=celoss(output,batch_image_type)
#             acc = accuracy(output, batch_image_type.unsqueeze(1))
#             print(j,"acc:",float(acc),"loss:",float(loss))
#     lr.step(epoch)

def train(train_loader:DataLoader, model:nn.Layer,loss_function, optimizer, epoch, args):
    """
    训练模型
    """
    sumloss=0
    sumacc=0
    for bid, (batch_image,batch_image_type,batch_image_import_flag) in enumerate(train_loader):
        output=model(batch_image[0])
        loss=loss_function(output,batch_image_type.unsqueeze(1))
        acc = accuracy(output, batch_image_type.unsqueeze(1))
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        sumloss+=float(loss)
        avgloss=sumloss/(bid+1)
        sumacc+=float(acc)
        avgacc=sumacc/(bid+1)
    with LogWriter(logdir=args.logdir) as writer:
        # use `add_scalar` to record scalar values
        writer.add_scalar(tag="train/loss", step=epoch, value=avgloss)
        writer.add_scalar(tag="train/acc", step=epoch, value=avgacc)
    return {"loss":avgloss,"acc":avgacc,"epoch":epoch}

def eval(test_loader, model:nn.Layer,loss_function, epoch, args):
    """
    评估模型
    """
    sumloss=0
    sumacc=0
    for bid ,(batch_image,batch_image_type,batch_image_import_flag) in enumerate(test_loader):
        output=model(batch_image[0])
        loss=loss_function(output,batch_image_type)
        acc = accuracy(output, batch_image_type.unsqueeze(1))
        sumloss+=float(loss)
        avgloss=sumloss/(bid+1)
        
        sumacc+=float(acc)
        avgacc=sumacc/(bid+1)
    with LogWriter(logdir=args.logdir) as writer:
        # use `add_scalar` to record scalar values
        writer.add_scalar(tag="eval/acc", step=epoch, value=avgacc)
        writer.add_scalar(tag="eval/loss", step=epoch, value=avgloss)
    return {"loss":avgloss,"acc":avgacc,"epoch":epoch}

def get_dataloader(dataset_dir,expansion,args):
    # 获得数据loader
    train_data,test_data=pipline_data_mlp(dataset_dir=dataset_dir,expansion=expansion,test_size=args.test_size)
    # 按照比例划分train 和 test数据
    pin_transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485], std=[0.229]
            )
        ]
    )
    # 数据预处理
    train_dataset,test_dataset=MLPDataset(train_data,pin_transform),MLPDataset(test_data,pin_transform)
    train_loader=DataLoader(train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        batch_sampler=None,
        drop_last=False,
    )
    test_loader=DataLoader(test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        batch_sampler=None,
        drop_last=False,
    )
    return train_loader,test_loader

def checkpoint(model:nn.Layer,optimizer,model_info:dict,checkpoint_dir):
    save_prefix="epoch_{:03d}_".format(model_info["epoch"])
    model_path=os.path.join(checkpoint_dir, save_prefix+"model.pdparams")
    encoder_k_model_path=os.path.join(checkpoint_dir, save_prefix+"encoder_k_model.pdparams")
    encoder_q_model_path=os.path.join(checkpoint_dir, save_prefix+"encoder_q_model.pdparams")
    optimizer_path=os.path.join(checkpoint_dir, save_prefix+"optimizer.pdopt")
    paddle.save(model.state_dict(),model_path)
    paddle.save(model.encoder_model_K.state_dict(),encoder_k_model_path)
    paddle.save(model.encoder_model_Q.state_dict(),encoder_q_model_path)
    paddle.save(optimizer.state_dict(),optimizer_path)
def main():
    args = parser.parse_args() 
    if args.seed is not None:
        random.seed(args.seed)
        paddle.seed(args.seed)
        #cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.cpu :
        paddle.device.set_device('cpu')
        warnings.warn(
            "当前采用cpu训练模型，最好选用GPU进行训练模型，在CPU上训练速度会非常慢 "
        )
    else:
        paddle.device.set_device('gpu')

    # 初始化模型
    encoder_q_model,encoder_k_model=load_model(args.moco_model)# 加载对比模型作为backbone
    from __init__ import HackResNet
    #cls_model=WordImageSliceMLPCLS(encoder_model_k=HackResNet(num_classes=128),encoder_model_q=HackResNet(num_classes=128),freeze_flag=False)
    cls_model=WordImageSliceMLPCLS(encoder_model_k=encoder_q_model,encoder_model_q=encoder_k_model,freeze_flag=False)
    
    # 加载数据
    train_loader,test_loader=get_dataloader(dataset_dir= args.data,expansion=args.expansion,args=args)
    lr=MYLR(learning_rate=args.lr,cos=args.cos,verbose=True)
    optimizer = optim.SGD(
            learning_rate=lr,
            parameters=cls_model.parameters(),
            weight_decay=args.weight_decay,
     )# pytorch 对应的优化器是SGD
    # 模型训练和评估
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(args.start_epoch, args.epochs):
        train_info=train(train_loader, cls_model,loss_function, optimizer, epoch, args)
        print(train_info)
        lr.step(epoch)# 更新一下学习率
        with paddle.no_grad():
            eval(test_loader,cls_model,loss_function,epoch,args)
        if epoch>0 and epoch%args.checkpoint_steps==0:
            checkpoint(cls_model,optimizer,train_info,args.checkpoint)
#    if global_steps % args.checkpoint_steps == 0:
if __name__ == "__main__":
    main()
    
