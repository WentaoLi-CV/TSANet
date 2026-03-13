import torch
from pytorch_wavelets import DWTForward
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
import numbers
import torch.nn.functional as F
from model.transformer import TransformerBlock


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.current_fuse_weights = None  # 训练时会被动态更新

    def forward(self, x, res):

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        # 保存当前的fuse_weights到实例属性中
        self.current_fuse_weights = fuse_weights.detach()  # detach()避免跟踪梯度
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class ModalInteractionModule(nn.Module):
    def __init__(self, dim):
        super(ModalInteractionModule, self).__init__()
        self.spa_att = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(dim * 2, dim // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(dim // 2, dim * 2, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(dim * 2, dim, 3, 1, 1)
        self.down = Conv(dim, dim, kernel_size=3, stride=2, dilation=1, bias=False)

    def forward(self, glb, local, modal):
        spa_map = self.spa_att(local - glb)
        spa_res = glb * spa_map + local
        cat_f = torch.cat([spa_res, glb], 1)
        cha_res = self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f)
        modal = self.down(modal)
        out = cha_res + modal

        return out



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class MDAF(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type):
        super(MDAF, self).__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_1_3 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv1_2_3 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv2_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_1_2 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_1_3 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv2_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_2_3 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)
        attn_113 = self.conv1_1_3(x1)
        attn_121 = self.conv1_2_1(x1)
        attn_122 = self.conv1_2_2(x1)
        attn_123 = self.conv1_2_3(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)
        attn_213 = self.conv2_1_3(x2)
        attn_221 = self.conv2_2_1(x2)
        attn_222 = self.conv2_2_2(x2)
        attn_223 = self.conv2_2_3(x2)

        out1 = attn_111 + attn_112 + attn_113 +attn_121 + attn_122 + attn_123
        out2 = attn_211 + attn_212 + attn_213 +attn_221 + attn_222 + attn_223
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x1+x2

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.proj_q = nn.Conv2d(dim, dim, 1)
        self.proj_kv = nn.Conv2d(dim, dim, 1)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Conv2d(dim, dim, 1)  # output projection (可选)

    def forward(self, freq, spt):
        b, c, h, w = spt.shape
        q_in = self.proj_q(spt)  # (B, C, H, W)
        kv_in = self.proj_kv(freq)

        def to_seq(x):
            x = x.flatten(2)  # (B, C, L)
            x = x.transpose(1, 2)  # (B, L, C)
            return x

        q_seq = to_seq(q_in)
        kv_seq = to_seq(kv_in)

        out, _ = self.attn(q_seq, kv_seq, kv_seq, need_weights=False)
        out = self.norm(out)

        out = out.transpose(1, 2).view(b, c, h, w)
        out = self.proj_out(out)

        return out


class GlobalLocalCross(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super(GlobalLocalCross, self).__init__()
        self.cross = CrossAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1),
            nn.Dropout(drop),
        )

    def forward(self, freq, spt):
        spt = spt + self.cross(freq, spt)  # (b,c,h,w)
        spt = spt + self.ffn(spt)

        return spt


class Bconv(nn.Module):
    def __init__(self,ch_in,ch_out,k,s):
        '''
        :param ch_in: 输入通道数
        :param ch_out: 输出通道数
        :param k: 卷积核尺寸
        :param s: 步长
        :return:
        '''
        super(Bconv, self).__init__()
        self.conv=nn.Conv2d(ch_in,ch_out,k,s,padding=k//2)
        self.bn=nn.BatchNorm2d(ch_out)
        self.act=nn.SiLU()

    def forward(self,x):
        '''
        :param x: 输入
        :return:
        '''
        return self.act(self.bn(self.conv(x)))


class SppCSPC(nn.Module):
    def __init__(self,ch_in,ch_out):
        '''
        :param ch_in: 输入通道
        :param ch_out: 输出通道
        '''
        super(SppCSPC, self).__init__()
        #分支一
        self.conv1=nn.Sequential(
            Bconv(ch_in,ch_out,1,1),
            Bconv(ch_out,ch_out,3,1),
            Bconv(ch_out,ch_out,1,1)
        )
        #分支二（SPP）
        self.mp1=nn.MaxPool2d(5,1,5//2) #卷积核为5的池化
        self.mp2=nn.MaxPool2d(9,1,9//2) #卷积核为9的池化
        self.mp3=nn.MaxPool2d(13,1,13//2) #卷积核为13的池化

        #concat之后的卷积
        self.conv1_2=nn.Sequential(
            Bconv(4*ch_out,ch_out,1,1),
            Bconv(ch_out,ch_out,3,1)
        )


        #分支三
        self.conv3=Bconv(ch_in,ch_out,1,1)

        #此模块最后一层卷积
        self.conv4=Bconv(2*ch_out,ch_out,1,1)

    def forward(self,x):
        #分支一输出
        output1=self.conv1(x)

        #分支二池化层的各个输出
        mp_output1=self.mp1(output1)
        mp_output2=self.mp2(output1)
        mp_output3=self.mp3(output1)

        #合并以上并进行卷积
        result1=self.conv1_2(torch.cat((output1,mp_output1,mp_output2,mp_output3),dim=1))

        #分支三
        result2=self.conv3(x)

        return self.conv4(torch.cat((result1,result2),dim=1))


class LocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 window_size=8,
                 ):
        super().__init__()

        self.local = SppCSPC(dim, dim)
        # self.bam = BAM(gate_channel=dim)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local(x)

        out = self.pad_out(local)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class LocalBlock(nn.Module):
    expansion = 1
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8,C=0,H=0,W=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn =LocalAttention(dim,window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class multilocalBlock(nn.Module):
    expansion = 1
    def __init__(self,dim=256, drop_path=0., norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LocalAttention(dim, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.drop_path(self.norm2(x))

        return x


class FMS(nn.Module):
    def __init__(self, in_ch, num_heads=8, window_size=8, mlp_ratio=2, bias=True, LayerNorm_type=None,
                 num_blocks=2):
        super(FMS, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')  # haar db2 sym4
        self.down_g = Conv(in_ch, in_ch, kernel_size=3, stride=2, dilation=1, bias=False)
        self.down_l = Conv(in_ch, in_ch, kernel_size=3, stride=2, dilation=1, bias=False)

        self.glb = nn.Sequential(*[
            TransformerBlock(dim=in_ch, num_heads=num_heads, ffn_expansion_factor=mlp_ratio,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for i in range(num_blocks)])

        self.localb = multilocalBlock(dim=in_ch, window_size=window_size)
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_glb = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_local = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)

        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)

        glb = self.outconv_bn_relu_glb(self.glb(self.down_g(x)))
        local = self.outconv_bn_relu_local(self.localb(self.down_l(x)))

        return yL,yH,glb,local


class SFINet(nn.Module):
    def __init__(self, dec_dim=96, num_heads=8, window_size=8, mlp_ratio=2, bias=True, LayerNorm_type='WithBias',
                 num_blocks=1, skit_add=False):
        super().__init__()
        self.skit_add = skit_add
        self.fuseFeature_vis = FMS(in_ch=dec_dim, num_heads=num_heads, window_size=window_size,
                                   mlp_ratio=mlp_ratio, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks)
        self.fuseFeature_ir = FMS(in_ch=dec_dim, num_heads=num_heads, window_size=window_size,
                                  mlp_ratio=mlp_ratio, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks)

        self.MDAF_L_vi = GlobalLocalCross(dec_dim, num_heads=num_heads)
        self.MDAF_L_ir = GlobalLocalCross(dec_dim, num_heads=num_heads)
        self.MDAF_H_vi = GlobalLocalCross(dec_dim, num_heads=num_heads)
        self.MDAF_H_ir = GlobalLocalCross(dec_dim, num_heads=num_heads)

        self.WF1_vi = ModalInteractionModule(dim=dec_dim)
        self.WF1_ir = ModalInteractionModule(dim=dec_dim)

        self.WF1_fus = WF(in_channels=dec_dim, decode_channels=dec_dim)

        scale_factor = 2
        self.upsample = nn.Sequential(
            nn.Conv2d(dec_dim, dec_dim * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(dec_dim, dec_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dec_dim, dec_dim, 3, padding=1)
        )

    def forward(self, vis, ir):
        # vis ir (b,c,h,w)
        wave_L_vi, wave_H_vi, glb_vi, loc_vi = self.fuseFeature_vis(vis)  # (b,c,h/2,w/2)
        wave_L_ir, wave_H_ir, glb_ir, loc_ir = self.fuseFeature_ir(ir)

        glbal_vi = self.MDAF_L_vi(wave_L_vi, glb_ir)  # (b,c,h/2,w/2)
        local_vi = self.MDAF_H_vi(wave_H_vi, loc_ir)  # (b,c,h/2,w/2)
        glbal_ir = self.MDAF_L_ir(wave_L_ir, glb_vi)
        local_ir = self.MDAF_H_ir(wave_H_ir, loc_vi)

        fusgl_vi = self.WF1_vi(glbal_vi, local_vi, ir)
        fusgl_ir = self.WF1_ir(glbal_ir, local_ir, vis)

        fus = self.WF1_fus(fusgl_vi, fusgl_ir)
        fus = self.upsample(fus)
        if vis.shape[-1]==7:
            fus = F.interpolate(fus, size=(7, 7), mode='bicubic', align_corners=False)
        if self.skit_add:
            fus = fus + vis + ir

        return fus