"""
可以蹭GPU的乞丐版本的moco对比学习
蹭单卡
"""
#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from  paddle import fluid
import moco.builder
from data.dataset import WordImagePiceDataset
import moco.loader
import paddle
#import torch.backends.cudnn as cudnn

#import torch.distributed as dist
import paddle.distributed as dit
#import torch.multiprocessing as mp
#import paddle.

#import torch.nn as nn
import paddle.nn as nn 
#import torch.nn.parallel
import paddle.distributed.parallel
#import torch.optim
import paddle.optimizer as optim
from moco.resnetmodels import HackResNet
from paddle.io import DataLoader

#import torchvision.models as models
#import torchvision.transforms as transforms
import paddle.vision.transforms as transforms

# model_names = sorted(
#     name
#     for name in models.__dict__
#     if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
# ) # 暂不使用

parser = argparse.ArgumentParser(description="乞丐版Paddle V100 字符切割训练")
parser.add_argument("--data",type=str,default="tmp/project_ocrSentences", metavar="DIR", help="path to dataset,指向按行切割的图片的文件夹目录")
parser.add_argument(
    "-j",
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="百度 ai-studio开两个进程就卡了一笔",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
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
    default=[120, 160],
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
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")# paddle的学习率使用策略和pytorch不一样



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

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    #ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = paddle.device.cuda.device_count()

    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     args.world_size = ngpus_per_node * args.world_size
    #     # Use torch.multiprocessing.spawn to launch distributed processes: the
    #     # main_worker process function
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
    #     # Simply call main_worker function
    #     main_worker(args.gpu, ngpus_per_node, args)
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print("=> creating model '{}'".format("自定义的Resnet"))
    model = moco.builder.MoCo(
        HackResNet,
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp,
    )
    print(model)
    #自习阅读了原来的代码，发现竟然只支持，多卡分布式训练，无语了，看不起单卡的人
    # 图像大小是244 244，的非常小的一部分，转换一下也相当于4卡训练了
    criterion = nn.CrossEntropyLoss()

    #lr_opti = optim.lr.CosineAnnealingDecay()
    optimizer = optim.Momentum(
        args.lr,
        momentum=args.momentum,# pytorch momentum
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
    )# pytorch 对应的优化器是SGD
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # if args.gpu is None:
            #     checkpoint = torch.load(args.resume)
            # else:
            #     # Map model to be loaded to specified single gpu.
            #     loc = "cuda:{}".format(args.gpu)
            #     checkpoint = torch.load(args.resume, map_location=loc)
            model.set_state_dict(paddle.load(args.resume))
            # 暂时不考虑load优化器数据，看起来优化器数据量也非常大
            #args.start_epoch = checkpoint["epoch"]
            #model.load_state_dict(checkpoint["state_dict"])
            #optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, 
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    traindir = os.path.join(args.data)#先暂时把数据加载

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
    traindir = os.path.join(args.data)#先暂时把数据加载
    train_dataset = WordImagePiceDataset(
        traindir, transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        #pin_memory=True, paddle 没有过
        #sampler=None,
        drop_last=True,
    )

    for epoch in range(args.start_epoch, args.epochs):
        # 
        # self.gnet_scheduler = optim.lr.CosineAnnealingDecay(max_learning_rate=self.lr,total_steps=100,end_learning_rate=self.lr/1000.0 , verbose=True)
        # adjust_learning_rate(optimizer, epoch, args) 先跑起来再说吧

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)


def adjust_learning_rate(optimizer, epoch, args):
    # 需要把学习率写成一个类，然后再调用
    pass

def train(train_loader, model:nn.Layer, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        # train_loader会提个一个batch_size的图片，然后会做一次数据增强，然后在TwoCropsTransform中一份数据生成两份，变成不同的k和q。
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)# 其实应该是同一个batch
            images[1] = images[1].cuda(args.gpu, non_blocking=True)# 

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)# 第0位是特别像的，但是第0位之后应该是特别不像的。

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), images[0].size(0)) pytorch
        losses.update(loss.item(), images[0].shape[0])# 用来计算损失是否在变小
        top1.update(acc1[0], images[0].shape[0])
        top5.update(acc5[0], images[0].shape[0])

        # compute gradient and do SGD step
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    paddle.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**{"name":self.name,"val":float(self.val),"avg":float(self.avg)})


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        # 256 个image，从output，找到值最大的五个坐标
        pred = pred.t()
        #correct = (pred==target.reshape([1, -1]).expand_as(pred))# paddle 用来标记，大的五个中，是否包含target中的数值。
        correct=fluid.layers.equal(pred,target.reshape([1, -1]).expand_as(pred).astype("int64"))


        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape([-1]).astype('float').sum(0, keepdim=True)
            #res.append(correct_k.mul_(100.0 / batch_size))
            #fluid.layers.nn.mul(correct_k,paddle.to_tensor([100.0 / batch_size]))
            res.append(correct_k*(100.0 / batch_size))

        return res


if __name__ == "__main__":
    main()