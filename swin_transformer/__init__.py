import torch
import torch.nn as nn
from numpy import load
from os.path import join as pjoin
import numpy as np
import torchvision
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:  # 如果drop率为0 则相当于不drop，直接输出，或者是不在训练的时候，就不需要drop
        return x
    keep_prob = 1 - drop_prob  # 保留率
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1)  # 生成一个shape ,shape的维度与x的维度相同，且第一个维度为 batchsize 若x->[3,196,768] shape->[3,1,1]
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype,
                                           device=x.device)  # 产生一个随机的tensor 范围为[1-drop_prob,2-drop_prob]均匀分布
    random_tensor.floor_()  # 将范围为[1-drop_prob,2-drop_prob]均匀分布向下取整数 则只有 0 或 1

    """
    将输入先除以保留率，保证输出均值与输入均值相同，不改变输入输出的分布，
    再乘随机产生的tensor，是得有的batch输出不变，有的batch输出为0，输出为0即代表随机drop掉了这个batch走过的路径
    https://zhuanlan.zhihu.com/p/587835249?utm_oi=1099351223446147072&utm_psn=1581065857459023872 讲的清楚
    """
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, head_num=12, qkv_bias=True, drop_ratio=0.1, attention_drop=False, droppath=False,
                 attention_dropratio=0.1):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.drop_attention = nn.Dropout(p=attention_dropratio) if attention_drop else nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=qkv_bias),
            DropPath(drop_ratio) if droppath else nn.Dropout(p=drop_ratio)
        )
    def forward(self, x):
        batch_size, token_num, dim = x.shape  # token_num 也就是 cat(class_token,patch_dim)
        qkv = self.qkv(x)  # qkv-> [batchsize,patch_dim+1,3*embeddding_dim]
        qkv = qkv.reshape(batch_size, token_num, 3, self.head_num,
                          dim // self.head_num)  # 这里先把qkv分开，分为了3个维度，再把分开后的qkv按照头数分开
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 此处把QKV完全分开 qkv->[3,batch,head_num,token_num,dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # qkv[0] 即为q ->[batch,head_num,token_num,dim]

        # 这里attention好像必须用 q*k转置 不然效果不一样
        attention = (q @ k.transpose(-1, -2)) / (self.head_num ** 0.5)  # attention 为 q*k/根号下（num_head）
        attention = attention.softmax(dim=-1)  # 这里的softmax的dim为-1 是为了在每个token_num 对应的那个行内做softmax
        attention = self.drop_attention(attention)

        v_attention = attention @ v  # v_attention ->[batch,head_num,token_num,dim]
        out = v_attention.permute(0, 2, 1, 3)  # v_attention ->[batch,token_num,head_num,dim]
        out = out.reshape(batch_size, token_num, dim)  # 这里保证输入和输出一样

        out = self.to_out(out)  # 找到的所以代码都有一个这样的映射关系
        return out


class MLP(nn.Module):
    def __init__(self, dim=768, drop_ratio=0.1, activation=nn.GELU, droppath=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.activation = activation()
        self.dropout = nn.Dropout(p=drop_ratio)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.droppath = DropPath(drop_ratio) if droppath else nn.Dropout(p=drop_ratio)
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x=self.droppath(x)
        return x


class blocks(nn.Module):
    def __init__(self, dim=768, head_num=12, qkv_bias=True,
                 drop_ratio=0.1, attention_drop=False, activation=nn.GELU,
                 attention_dropratio=0.1
                 ):
        super(blocks, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.MHA = MultiHeadAttention(
            dim=dim,
            head_num=head_num,
            qkv_bias=qkv_bias,
            drop_ratio=drop_ratio,
            attention_drop=attention_drop,
            attention_dropratio=attention_dropratio
        )
        self.mlp_channels = MLP(
            dim=dim,
            drop_ratio=drop_ratio,
            activation=activation
        )
        self.norm2 = nn.LayerNorm(dim)


    def forward(self, x):
        x = x + self.MHA(self.norm1(x))
        x = x + self.mlp_channels(self.norm2(x))

        return x




class MLP_Head(nn.Module):
    def __init__(self, dim=768, num_class=1000, prelogit=True):
        super(MLP_Head, self).__init__()
        self.prelogit = nn.Identity() if prelogit else \
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh()
            )
        self.linear = nn.Linear(dim, num_class)

    def forward(self, x):
        x = self.linear(self.prelogit(x))
        return x


class Vit(nn.Module):
    def __init__(self, dim=768, head_num=12, qkv_bias=True,
                 patch_size=16, attention_drop=False,
                 droppath=True, drop_ratio=0.1,
                 activation=nn.GELU,
                 depth=12, attention_dropratio=0.1,
                 num_class=1000, prelogit=False
                 ):
        super(Vit, self).__init__()

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, (patch_size, patch_size), stride=(patch_size, patch_size)),
            nn.Flatten(start_dim=2)  # pytorch input_img -> (batch size, image channel, image height, image width)
        )

        self.cls_token = nn.Parameter(torch.randn(1, dim, 1))

        assert 224 % patch_size == 0, '224/Patch_Size must be integer'

        self.position_embedding = nn.Parameter(torch.randn(1, dim, (224 // patch_size) ** 2 + 1))

        self.position_dropout = nn.Dropout(drop_ratio)

        self.dropratio = [x.item() for x in torch.linspace(0, drop_ratio, depth)]

        self.blocks = nn.Sequential(  # 这里随着层数的变深，使drop_ratio率变高
            *[blocks(
                dim=dim, head_num=head_num, qkv_bias=qkv_bias,
                drop_ratio=self.dropratio[i], attention_drop=attention_drop,
                activation=activation, attention_dropratio=attention_dropratio
            )
                for i in range(depth)
            ]
        )
        self.layernorm = nn.LayerNorm(dim)

        self.mlphead = MLP_Head(
            dim=dim, num_class=num_class, prelogit=prelogit
        )

    def forward(self, x):
        x = self.patch_embedding(x)

        cls_token = self.cls_token.expand(x.shape[0], -1,
                                          -1)  # cls_token作为参数为 [1，768，1]，直接cat 无法拼接，产生一个中间量，把第零维变为batchsize

        x = torch.cat((cls_token, x), dim=-1)
        x = self.position_embedding + x
        x = self.position_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.blocks(x)

        x = self.layernorm(x)

        x = x[:, 0]  # 去除class_token

        x = self.mlphead(x)

        x = x.softmax(dim=-1)

        return x


def vit_base_patch16_224(num_classes=1000):
    model = Vit(dim=768, head_num=12, qkv_bias=True,
                patch_size=16, attention_drop=False,
                droppath=False, drop_ratio=0.1,
                activation=nn.GELU,
                depth=12, attention_dropratio=0.1,
                num_class=num_classes,prelogit=False)
    return model


def vit_base_patch16_224_in21k(num_classes=21843, prelogit=True):
    model = Vit(dim=768, head_num=8, qkv_bias=True,
                patch_size=16, attention_drop=False,
                droppath=True, drop_ratio=0.1,
                activation=nn.GELU,
                depth=12, attention_dropratio=0.1,
                num_class=num_classes, prelogit=prelogit)
    return model


def vit_base_patch32_224(num_class=1000):
    model = Vit(dim=768, head_num=12, qkv_bias=True,
                patch_size=32, attention_drop=False,
                droppath=True, drop_ratio=0.1,
                activation=nn.GELU,
                depth=12, attention_dropratio=0.1,
                num_class=num_class)

    return model


def vit_base_patch32_224_in21k(num_class=1000, prelogit=True):
    model = Vit(dim=768, head_num=12, qkv_bias=True,
                patch_size=32, attention_drop=False,
                droppath=True, drop_ratio=0.1,
                activation=nn.GELU,
                depth=12, attention_dropratio=0.1,
                num_class=num_class, prelogit=prelogit)
    return model


if __name__ == '__main__':
    model = vit_base_patch16_224()
    print(model)
    out = model(torch.rand(3, 3, 224, 224))
    out_1 = out[0]
    out_2 = out[1]
    sum_1 = sum(out_1)
    sum_2 = sum(out_2)
    print(1)
