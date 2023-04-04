import os 
import sys 
PROJECT_DIR= os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.realpath( __file__))
                                    )
                                )
                            )
sys.path.append(PROJECT_DIR)
sys.path.append(PROJECT_DIR+"/moco")
from paddle import nn 
import paddle
class WordImageSliceMLPCLS(nn.Layer):
    """
    基于神经网络的0-1分类器
    """
    def __init__(self,encoder_model:nn.Layer,dim=128,num_classes=2) -> None:
        super().__init__()
        self.encoder_model:nn.Layer=encoder_model #基于对比学习的backbone 
        # 冻结backbone的参数
        for param in encoder_model.all_parameters():
            param.stop_gradient = True
        self.linear = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 8), 
                nn.ReLU(),
                nn.Linear(8, num_classes)
            )
    
    def forward(self, *inputs, **kwargs):
        x = self.encoder_model(inputs)
        x=self.linear(x)
        return x



if __name__ == "__main__":
    #main()
    from moco.pp_infer import load_model
    encoder_q_model,encoder_k_model=load_model("tmp/checkpoint/epoch_011_bitchth_003500_model.pdparams")
    mlpcls=WordImageSliceMLPCLS(encoder_k_model)
    params_info = paddle.summary(mlpcls,(1, 1, 16, 48))
    print(params_info)