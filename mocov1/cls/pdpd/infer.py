
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
    # encoder_k_model.set_state_dict(paddle.load("tmp/checkpoint/epoch_105_encoder_k_model.pdparams"))
    # encoder_q_model.set_state_dict(paddle.load("tmp/checkpoint/epoch_105_encoder_q_model.pdparams"))
    encoder_k_model.set_state_dict(paddle.load("tmp/nobackbone/epoch_030_encoder_k_model.pdparams"))
    encoder_q_model.set_state_dict(paddle.load("tmp/nobackbone/epoch_030_encoder_k_model.pdparams"))
    cls_model=WordImageSliceMLPCLS(encoder_model_k=encoder_k_model,encoder_model_q=encoder_q_model,freeze_flag=True)
    cls_model.set_state_dict(paddle.load("tmp/nobackbone/epoch_030_model.pdparams"))
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

def fast_infer():
    from paddle.vision import transforms
    from paddle.io import DataLoader 
    from  mocov1.pp_infer import WIPDataset
    from mocov1.moco.loader import TwoCropsTransform
    from PIL import Image
    from mocov1.render import render_html
    # -------------------------------
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
    model_ds=WIPDataset(data_dir="tmp/project_ocrSentences_dataset/195",transform=TwoCropsTransform(transforms.Compose(augmentation)))#这个是模型使用，要对数据做一些变化。
    train_loader = DataLoader(
            model_ds,
            batch_size=256,
            shuffle=False,
            num_workers=1,
            #pin_memory=True, paddle 没有过
            #sampler=None,
            drop_last=False,
        )
    cls_model=load_model()
    import glob, os
    for f in glob.glob("tmp/project_ocrSentences_dataset/word_image_slice/word_s*.png"):
        os.remove(f)
    image_index=0
    for k, (images, _) in enumerate(train_loader):    
        predict_info=cls_model(images[0])

        predict_labels=paddle.argmax(predict_info,axis=-1)
        seg_img_numpy=images[2].numpy()# 这个图片是原始的图片
        img_len=seg_img_numpy.shape[0]
        i=0
        while i< img_len:
            # 
            # axes=plt.subplot(4,24,j)
            # axes.set_title(str(predict_info[i])+"->"+str(i))
            seg_img=seg_img_numpy[i,:,:]
            h,w=seg_img.shape
            seg_img[:,w//2]=0
            #plt.imshow(seg_img )
            pil_image=Image.fromarray(seg_img)
            pil_image.save("tmp/project_ocrSentences_dataset/word_image_slice/word_seg_{:04d}_type_{:02d}.png".format(image_index,int(predict_labels[i])))
            image_index+=1
            i+=1
    render_html()
def main():
    args = parser.parse_args()
    test_infer(args) 
    #cls_model=load_model()


def infer_single_image(image_byte):
    """
    输入一张图片：
    输出：基于图片的输出分割线。
    """
    pass 

if __name__ == '__main__':
    with paddle.no_grad():
        #load_dataset_from_image()
        fast_infer()
