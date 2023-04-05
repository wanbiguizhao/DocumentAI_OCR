
import argparse
import paddle 
from paddle import nn
from __init__ import HackResNet
from models import WordImageSliceMLPCLS
from train import get_dataloader
from paddle.metric import accuracy

parser = argparse.ArgumentParser(description="推测一个神经网络")
parser.add_argument("--data",type=str,default="mocov1/dataset", metavar="DIR", help="path to dataset,指向按行切割的图片的文件夹目录")
parser.add_argument(
    "--expansion",
    default=1,
    type=int,
    metavar="EXPAN",
    help="对于连着的图片的数据，复制倍数",
)
parser.add_argument(
    "--test-size",
    default=0.01,
    type=float,
    metavar="TS",
    help="test size of all data ",
    #dest="lr",
)
parser.add_argument(
    "-j",
    "--workers",
    default=2,
    type=int,
    metavar="N",
    help="加载dataset的work",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help=" 这个意思莫非是，多卡GPU的情况，256被多卡平均使用，想多了，用不到多卡",
)
parser.add_argument("--cpu", action="store_true", help="使用cpu训练")# paddle的学习率使用策略和pytorch不一样

def load_model():
    # 这块先进行硬编码把
    encoder_k_model=HackResNet(num_classes=128)
    encoder_q_model=HackResNet(num_classes=128)
    encoder_k_model.set_state_dict(paddle.load("tmp/checkpoint/epoch_105_encoder_k_model.pdparams"))
    encoder_q_model.set_state_dict(paddle.load("tmp/checkpoint/epoch_105_encoder_q_model.pdparams"))
    cls_model=WordImageSliceMLPCLS(encoder_model_k=encoder_k_model,encoder_model_q=encoder_q_model,freeze_flag=True)
    cls_model.set_state_dict(paddle.load("tmp/checkpoint/epoch_105_model.pdparams"))
    return cls_model

def test_infer(args):
    cls_model=load_model()
    cls_model.eval()
    train_loader,test_loader=get_dataloader(dataset_dir=args.data,expansion=1,args=args)
    with paddle.no_grad():
        sumacc=0
        for bid, (batch_image,batch_image_type,batch_image_import_flag) in enumerate(train_loader):
            output=cls_model(batch_image[0])
            acc = accuracy(output, batch_image_type.unsqueeze(1))
            sumacc+=float(acc)
            avgacc=sumacc/(bid+1)
            print(avgacc)


def main():
    args = parser.parse_args()
    test_infer(args) 
    #cls_model=load_model()

if __name__ == '__main__':
    with paddle.no_grad():
        #load_dataset_from_image()
        main()
