import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import fusion_prompt_loss
from model.sa_modal import ImageTextFusSam
from model.fus_model import SFINet
from model.transformer import TransformerBlock, Downsample, Upsample


class HybridCconsistencyLoss(nn.Module):
    def __init__(self):
        super(HybridCconsistencyLoss, self).__init__()

    def forward(self, feat1, feat2, alpha=0.5, beta=0.5):
        mse = F.mse_loss(feat1, feat2)
        cos_loss = 1 - F.cosine_similarity(feat1, feat2, dim=-1).mean()

        return alpha * mse + beta * cos_loss


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class FusionEmb(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(FusionEmb, self).__init__()
        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x


class WeightModule(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 1),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


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
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TextGuide(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)

        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)

        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, fea_img, fea_text):
        batch, channel, height, width = fea_img.shape

        n_head = self.n_head
        head_dim = channel // n_head

        fea_img = self.norm_A(fea_img)
        qkv_img = self.qkv_A(fea_img).view(batch, n_head, head_dim * 3, height, width)
        query_img, _, _ = qkv_img.chunk(3, dim=2)

        fea_text = self.norm_B(fea_text)
        qkv_text = self.qkv_B(fea_text).view(batch, n_head, head_dim * 3, height, width)
        _, key_text, value_text = qkv_text.chunk(3, dim=2)

        attn_img = torch.einsum("bnchw, bncyx -> bnhwyx", query_img, key_text).contiguous() / math.sqrt(channel)
        attn_img = attn_img.view(batch, n_head, height, width, -1)
        attn_img = torch.softmax(attn_img, -1)
        attn_img = attn_img.view(batch, n_head, height, width, height, width)

        out_img = torch.einsum("bnhwyx, bncyx -> bnchw", attn_img, value_text).contiguous()
        out_img = self.out_B(out_img.view(batch, channel, height, width))
        out_img = out_img + fea_text

        return out_img


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ImageTextFus(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(ImageTextFus, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super().__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(num_blocks)):
            embed_dim = int(dim * (2 ** i))

            self.levels.append(nn.Sequential(*[
                TransformerBlock(dim=embed_dim, num_heads=heads[i],
                                 ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type)
                for _ in range(num_blocks[i])
            ]))

            if i < len(num_blocks) - 1:
                self.downsamples.append(Downsample(embed_dim))

    def forward(self, x, multiout=False):
        x = self.patch_embed(x)
        features = []

        for i in range(len(self.levels)):
            x = self.levels[i](x)
            features.append(x)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        if multiout:
            return features[::-1]
        else:
            return features[-1]


class MultiModel(nn.Module):

    def __init__(self, model_clip, inp_dim_vis=3, inp_dim_ir=3, out_chans=3, dim=48, num_blocks=[2, 2, 2, 2],
                 decvit_depth=4, heads=[1, 2, 4, 8], mlp_ratio=2, bias=False, LayerNorm_type='WithBias',
                 w_consist=0.1, w_fus=1, patch_size=4, fea_size=96, skit_add=False, num_fus=[2, 2, 2, 2]):
        super().__init__()
        self.skit_add = skit_add
        self.w_consist = w_consist
        self.w_fus = w_fus

        # 1. Encoder
        self.model_clip = model_clip
        self.model_clip.eval()

        self.encoder_vis = FeatureEncoder(inp_channels=inp_dim_vis, dim=dim, num_blocks=num_blocks, heads=heads,
                                          ffn_expansion_factor=mlp_ratio, bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_ir = FeatureEncoder(inp_channels=inp_dim_ir, dim=dim, num_blocks=num_blocks, heads=heads,
                                         ffn_expansion_factor=mlp_ratio, bias=bias, LayerNorm_type=LayerNorm_type)

        # 2. Decoder
        self.imgtextfus_vi = nn.ModuleList()
        self.imgtextfus_ir = nn.ModuleList()
        self.fusup = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in range(4):
            embed_dim = int(dim * (2 ** i))
            img_emb_size = (fea_size / (2 ** i), fea_size / (2 ** i))

            self.imgtextfus_vi.append(
                ImageTextFusSam(dim=embed_dim, text_channel=512, image_embedding_size=img_emb_size))
            self.imgtextfus_ir.append(
                ImageTextFusSam(dim=embed_dim, text_channel=512, image_embedding_size=img_emb_size))

            self.decoder.append(nn.Sequential(*[
                TransformerBlock(dim=embed_dim, num_heads=heads[i], ffn_expansion_factor=mlp_ratio,
                                 bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[i])
            ]))

            if i == 3:
                self.textguide_vi = TextGuide(embed_dim)
                self.textguide_ir = TextGuide(embed_dim)
                self.fus4 = SFINet(dec_dim=embed_dim, num_heads=heads[i], window_size=8, mlp_ratio=mlp_ratio,
                                   bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_fus[i], skit_add=skit_add)
                self.upsample.append(Upsample(embed_dim))
                self.fusup.append(nn.Identity())

            else:
                self.fusup.append(FusionEmb(embed_dim=embed_dim))
                if i > 0:
                    self.upsample.append(Upsample(embed_dim))
                else:
                    self.upsample.append(nn.Identity())

        # 3. Out & Loss
        self.decoder_out = nn.Sequential(*[
            TransformerBlock(dim=int(dim * (2 ** 0)), num_heads=heads[0], ffn_expansion_factor=mlp_ratio,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(decvit_depth)
        ])
        self.out = nn.Conv2d(int(dim * (2 ** 0)), out_chans, kernel_size=3, stride=1, padding=1, bias=bias)

        self.consistency = HybridCconsistencyLoss()
        self.fusloss = fusion_prompt_loss()

    def forward(self, vis, ir, text_vis, text_ir, return_visuals=False):
        # Text Encoder
        with torch.no_grad():
            fea_text_vis = self.model_clip.encode_text(text_vis).to(vis.dtype)
            fea_text_ir = self.model_clip.encode_text(text_ir).to(ir.dtype)

        # Image Encoder
        fea_vis4, fea_vis3, fea_vis2, fea_vis1 = self.encoder_vis(vis, multiout=True)
        fea_ir4, fea_ir3, fea_ir2, fea_ir1 = self.encoder_ir(ir, multiout=True)

        fea_vis_list = [fea_vis1, fea_vis2, fea_vis3, fea_vis4]
        fea_ir_list = [fea_ir1, fea_ir2, fea_ir3, fea_ir4]

        # Level 4
        fea_imgtext_vi4 = self.imgtextfus_vi[3](fea_vis_list[3], fea_text_vis)
        fea_imgtext_ir4 = self.imgtextfus_ir[3](fea_ir_list[3], fea_text_ir)
        fea_guide_vi4 = self.textguide_vi(fea_vis_list[3], fea_imgtext_vi4)
        fea_guide_ir4 = self.textguide_ir(fea_ir_list[3], fea_imgtext_ir4)

        fea_fus = self.fus4(fea_imgtext_vi4, fea_imgtext_ir4)
        fea_fus4 = self.decoder[3](fea_fus)
        fea_fus_up = self.upsample[3](fea_fus4)

        # Level 3 - Level 1
        for i in range(2, -1, -1):
            fea_imgtext_vi = self.imgtextfus_vi[i](fea_vis_list[i], fea_text_vis)
            fea_imgtext_ir = self.imgtextfus_ir[i](fea_ir_list[i], fea_text_ir)

            fea_fus = fea_imgtext_vi + fea_imgtext_ir
            fea_fus = self.fusup[i](fea_fus, fea_fus_up)
            fea_fus = self.decoder[i](fea_fus)

            if i > 0:
                fea_fus_up = self.upsample[i](fea_fus)
            else:
                fea_fus_up = fea_fus

        # Out
        fea_fus_out = self.decoder_out(fea_fus_up)
        fus_img = self.out(fea_fus_out)

        # Loss
        consistency_loss_vi4 = self.consistency(fea_fus4, fea_guide_vi4)
        consistency_loss_ir4 = self.consistency(fea_fus4, fea_guide_ir4)
        loss_consistency = consistency_loss_vi4 + consistency_loss_ir4

        total_loss_base, loss_ssim, loss_max, loss_color, loss_text = self.fusloss(vis, ir, fus_img)
        total_loss = self.w_consist * loss_consistency + self.w_fus * total_loss_base

        if return_visuals:
            visual_dict = {
                "fea_vis1": fea_vis1,
                "fea_ir1": fea_ir1,
                "fea_imgtext_vi1": self.imgtextfus_vi[0](fea_vis_list[0], fea_text_vis),
                "fea_imgtext_ir1": self.imgtextfus_ir[0](fea_ir_list[0], fea_text_ir),
            }
            return (total_loss, loss_ssim, loss_max, loss_color, loss_text, loss_consistency), fus_img, visual_dict

        return (total_loss, loss_ssim, loss_max, loss_color, loss_text, loss_consistency), fus_img

