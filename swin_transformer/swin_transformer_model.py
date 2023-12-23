import torch
import torch.nn as nn
from Vit.vit_model import vit_base_patch16_224


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


def window_partition(x, window_size):
    """
    输入
    x->[batch,h,w,c]
    把不同window放入batch中，再做attention

    输出
    return [batch*window_num,window_size,window_size,c]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(x, window_size, H, W):
    """
       输入
       x->[batch*window_num,window_size,window_size,c]

       输出
       return [batch,H,W,C]
       """
    B = int(x.shape[0] // ((H / window_size) * (W / window_size)))
    x = x.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, input_dim, window_size, num_heads, drop_ratio, qkv_bias=True,attn_drop=0.):
        super().__init__()

        self.qkv = nn.Linear(input_dim, 3 * input_dim, bias=qkv_bias)
        self.window_size = window_size
        self.num_heads = num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        self.softmax = nn.Softmax(dim=-1)

        self.attn_drop=nn.Dropout(attn_drop)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(drop_ratio)

        """
        这里 self.scale 就是dk
        dk 作用是为了让 Q*K_t 之后 概率分布不变 
        假设 Q K 独立 且 Q ~ N(0,1)  K~ N(0,1)
         E（qi*ki）=0   Var（qi*ki）= 1
         Σ E(qi*ki)=0  Σ Var(qi*ki)=dk ->这里相当于 Q为[token_num,dim] dim=dk 这里的Q为分为多头之后的Q
         所以 在分多头之前 Q,K,V ->[token_num,input_dim]  再分为多个头的Q,K,V ->[token_num,input_dim/num_head]
         即input_dim/num_head=dk
         
        """

        self.scale = (input_dim//self.num_heads) ** -0.5
        # 以下产生相对位置编码索引 index
        code_H = torch.arange(self.window_size)
        code_W = torch.arange(self.window_size)

        code_mesh = torch.stack(torch.meshgrid([code_H, code_W]))  # code_mesh-> [2,window_size,window_size] 可以看作绝对位置索引
        code_mesh_flatten = torch.flatten(code_mesh, 1)  # code_mesh_flatten -> [2,window_size*window_size]

        relative_position_code = code_mesh_flatten.unsqueeze(2) - code_mesh_flatten.unsqueeze(
            1)  # 这里等效 两个[2,49,49] 矩阵相减，只不过方向不同
        relative_position_code_bias = relative_position_code + (self.window_size - 1) * torch.ones_like(
            relative_position_code)  # 转换到最小从0开始
        relative_position_code_bias[0] = relative_position_code_bias[0] * (2 * self.window_size - 1)
        relative_position_index = relative_position_code_bias[0] + relative_position_code_bias[1]


        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        """
        输入的x是已经分过window之后,且已经展平成为token的x ->[batch*window_num,n,c]
        在shifted window 时需要用到mask ->[window_num, token_num, token_num]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # qkv[0] 即为q ->[batch*window_num,head_num,token_num,dim]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        atten = (q @ k.transpose(-1, -2)) * self.scale + relative_position_bias.unsqueeze(
            0)  # atten ->[batch*window_num,head_num,token_num,token_num]




        if mask is not None:
            window_num = mask.shape[0]

            mask = mask.unsqueeze(1).unsqueeze(0)  # mask ->[1,window_num,1,token_num,token_num]
            atten = atten.view(B // window_num, window_num, self.num_heads, N,
                               N)  # atten ->[batch,window_num,head_num,token_num,token_num]
            atten = atten + mask
            atten = atten.view(-1, self.num_heads, N, N)
            atten = self.softmax(atten)

        else:
            atten = self.softmax(atten)



        atten=self.attn_drop(atten)
        """
        out = (atten @ v).view(B, N, C) 结果全奔溃 原因是 (atten @ v) 结果为 [batch,num_heads,token_num,token]
        
        需要先做一个transpose 变成[batch,token_num,num_heads，token] 
        再把 num_heads 放入token中即多个头合成一个头
        
        原论文代码这样做的
        out=(atten @ v).transpose(1,2).reshape(B, N, C)
        """
        out = (atten @ v).permute(0,2,1,3).contiguous().view(B,N,C)
        # out = (atten @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_feature, hidden_feature=None, out_feature=None, act_lyaer=nn.GELU, drop_ratio=0):
        super().__init__()
        out_feature = in_feature or out_feature
        hidden_feature = hidden_feature or in_feature
        self.fc1 = nn.Linear(in_feature, hidden_feature)
        self.act = act_lyaer()
        self.fc2 = nn.Linear(hidden_feature, out_feature)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        out = self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
        return out


class SwinTransformerBlock(nn.Module):
    def __init__(self, input_dim, input_size,
                 num_head, shift_size,
                 window_size, atten_drop_ratio,
                 mlp_drop_ratio, qkv_bias=True, mlp_ratio=4.,
                 drop_path=0.
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.input_dim = input_dim
        self.shift_size = shift_size
        self.input_size = input_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_head
        self.atten_drop = atten_drop_ratio
        self.mlp_drop = mlp_drop_ratio

        self.norm2 = nn.LayerNorm(input_dim)

        if self.input_size <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_size

        hidden_feature = int(input_dim * mlp_ratio)
        self.mlp = MLP(
            in_feature=input_dim, hidden_feature=hidden_feature,
            act_lyaer=nn.GELU, drop_ratio=0
        )

        self.drop_path = nn.Dropout(drop_path)
        self.attn = WindowAttention(input_dim, self.window_size, self.num_heads, self.atten_drop, qkv_bias=qkv_bias)
        """
        atten_mask ->[batch_size*num_window,token_num,token_num]
        atten_mask 分为4部分
        """
        if self.shift_size > 0:
            img_mask = torch.zeros(1, self.input_size, self.input_size, 1)

            slice_H = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size)
                       )
            slice_W = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size)
                       )

            cnt = 0
            for h in slice_H:
                for w in slice_W:
                    img_mask[:, h, w, :] = cnt
                    cnt = cnt + 1

            window_mask = window_partition(img_mask,
                                           self.window_size)  # 分成一个一个window window_mask-> [batch*window_num,window_size,window_size]
            window_mask_flatten = window_mask.view(-1, self.window_size * self.window_size)
            atten_mask = window_mask_flatten.unsqueeze(1) - window_mask_flatten.unsqueeze(2)
            atten_mask = atten_mask.masked_fill(atten_mask != 0, float(-100.0)).masked_fill(atten_mask == 0, float(0.0))

        else:
            atten_mask = None

        self.register_buffer("attn_mask", atten_mask)

    def forward(self, x):
        """
        此处输入的x是 [Batch,H*W,C]
        """
        B, token_num, C = x.shape  # token_num = self.input_size * self.input_size

        shortcut = x
        x = self.norm1(x)

        x = x.view(B, self.input_size, self.input_size,C)  # 此处将 x 变成图片形式

        if self.shift_size > 0:
            x_shift = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_window = window_partition(x_shift, self.window_size)

        else:
            x_shift = x
            x_window = window_partition(x_shift, self.window_size)

        x_window = x_window.view(-1, self.window_size * self.window_size, C)

        x_attn = self.attn(x_window, mask=self.attn_mask)
        x_attn = x_attn.view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            x_reverse = window_reverse(x_attn, self.window_size, self.input_size,
                                       self.input_size)  # 输出放成图片 [Batch,h,w,c]
            x_shift_back = torch.roll(x_reverse, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        else:
            x_reverse = window_reverse(x_attn, self.window_size, self.input_size,
                                       self.input_size)  # 输出放成图片 [Batch,h,w,c]
            x_shift_back = x_reverse

        x = x_shift_back.view(B, self.input_size * self.input_size, C)

        x = shortcut + self.drop_path(x)

        out = x + self.drop_path(self.mlp(self.norm2(x)))

        return out


class PatchMerging(nn.Module):
    def __init__(self, input_dim, input_size):
        super().__init__()

        self.input_size = input_size

        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)

    def forward(self, x):
        B, N, C = x.shape

        x = x.view(B, self.input_size, self.input_size, C)
        """
        以下对x进行抽样 分成4个channel
        """
        x_1 = x[:, 0::2, 0::2, :]
        x_2 = x[:, 1::2, 0::2, :]
        x_3 = x[:, 0::2, 1::2, :]
        x_4 = x[:, 1::2, 1::2, :]

        x = torch.cat([x_1, x_2, x_3, x_4], -1)  # 此时 x的C 为输入出时候的4倍

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        out = self.reduction(x)

        return out


"""" 
    embed_dim=96 -> swin-t swin-s
    embed_dim=128 -> swin-b
    embed_dim=192 ->swin-L
"""
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, input_dim=3, patch_size=4, embed_dim=128, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.patch_resolution = img_size // patch_size
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        if norm_layer:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        assert x.shape[2] == self.img_size and x.shape[3] == self.img_size, \
            f"input img size must be ({self.img_size}*{self.img_size}) "

        # x_size (batch,3,h,w) -> (batch,embed_dim,h*w) -> (batch,h*w,embed_dim) 将embed_dim放在最后目的是为了layernorm
        x = self.proj(x).flatten(2).transpose(1, 2)

        if self.norm:
            x = self.norm(x)
        else:
            return x
        return x


class BaseLayer(nn.Module):
    def __init__(self, input_dim, input_size, depth, num_head,
                 window_size, atten_drop_ratio,
                 mlp_drop_ratio, qkv_bias=True, mlp_ratio=4.,
                 drop_path=0., downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    input_dim=input_dim, input_size=input_size,
                    num_head=num_head, shift_size=0 if i % 2 == 0 else window_size // 2,
                    window_size=window_size, atten_drop_ratio=atten_drop_ratio,
                    mlp_drop_ratio=mlp_drop_ratio, qkv_bias=qkv_bias, mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
                )
                for i in range(depth)
            ]
        )
        if downsample is not None:
            self.downsample = PatchMerging(
                input_size=input_size,
                input_dim=input_dim
            )
        else:
            self.downsample = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Swin_Transformer(nn.Module):
    def __init__(self,
                 img_size=224, input_dim=3, num_classes=1000,
                 patch_size=4, embed_dim=128,
                 depth=[2, 2, 18, 2], num_head=[4, 8, 16, 32],
                 norm_layer=nn.LayerNorm, window_size=7, atten_drop_ratio=0.,
                 mlp_drop_ratio=0., qkv_bias=True, mlp_ratio=4., drop_ratio=0.,
                 drop_path=0.,drop_path_rate=0.1):
        super(Swin_Transformer, self).__init__()

        self.num_stage = len(depth)
        self.num_features = int(embed_dim * 2 ** (self.num_stage - 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            input_dim=input_dim,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        self.patch_embed_size = self.patch_embed.patch_resolution
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_stage):
            layer = BaseLayer(
                input_dim=embed_dim * (2 ** i_layer),
                input_size=self.patch_embed_size // (2 ** i_layer),
                depth=depth[i_layer],
                num_head=num_head[i_layer],
                window_size=window_size,
                atten_drop_ratio=atten_drop_ratio,
                mlp_drop_ratio=mlp_drop_ratio,
                qkv_bias=qkv_bias,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depth[:i_layer]):sum(depth[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_stage - 1) else None
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


def Swin_Transformer_b_patch4_window7_224():
    model_swin = Swin_Transformer()
    return model_swin


# if __name__ == '__main__':
#     model_vit = vit_base_patch16_224()
#     model_dic = model_vit.state_dict()
#
#     model_swin = Swin_Transformer()
#     model_swin_dic = model_swin.state_dict()
#     weight = torch.load("D:\Jonior_Learn\swin_base_patch4_window7_224.pth")
#     w = weight["model"]
#     index = w["layers.2.blocks.15.attn.relative_position_index"]
#
#     print(1)
