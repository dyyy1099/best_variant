from torch import nn
import torch
from models.Temporal_Model import *
import torchaudio
import math
from AudioMAE import audio_models_vit
from timm.models.layers import to_2tuple
import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
        gs_old=None,
) -> torch.Tensor:
    # 保持位置嵌入调整逻辑不变
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    if gs_old is None:
        gs_old = (int(math.sqrt(len(posemb_grid))), int(math.sqrt(len(posemb_grid))))

    if gs_new is None or not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old[0], gs_old[1], -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


class PatchEmbed_new(nn.Module):
    '''音频特征Patch嵌入（复用原有逻辑）'''
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

        _, _, h, w = self.get_output_shape(img_size)
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class GenerateModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 时序网络保持不变（输入维度512）
        self.temporal_net = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=512,
            depth=args.temporal_layers,
            heads=8,
            mlp_dim=1024,
            dim_head=64
        )

        # 分类器（回归输出）
        self.our_classifier = torch.nn.Linear(512, 1)
        # 音频特征投影层（768→512，匹配时序网络输入）
        self.audio_proj = torch.nn.Linear(768, 512)

        # 仅保留音频相关参数
        self.n_audio = 256
        self.n_progr = 3  # 可学习提示词数量

        # 仅构建音频模型（删除图像模型）
        self._build_audio_model()

    def _build_audio_model(self, model_name='vit_base_patch16', drop_path_rate=0.1, global_pool=False, mask_2d=True,
                           use_custom_patch=False, ckpt_path='./audiomae_pretrained.pth'):
        self.audio_model = audio_models_vit.__dict__[model_name](
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
            mask_2d=mask_2d,
            use_custom_patch=use_custom_patch,
            n_seq=self.n_audio,
            n_progr=self.n_progr
        )
        # 加载预训练权重
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt = ckpt['model']
        orig_pos_embed = ckpt['pos_embed']
        print(orig_pos_embed.shape, self.audio_model.pos_embed.shape)
        new_posemb = resize_pos_embed(
            orig_pos_embed, 
            self.audio_model.pos_embed, 
            gs_old=(1024 // 16, 128 // 16),
            gs_new=(512 // 16, 128 // 16)
        )
        ckpt['pos_embed'] = new_posemb

        # 初始化位置嵌入（包含提示词）
        emb = torch.randn(1, self.n_audio + self.n_progr * (len(self.audio_model.blocks) // 6) + 1, 768)
        emb[:, :self.n_audio + 1] = ckpt['pos_embed'][:, :self.n_audio + 1]
        del ckpt['pos_embed']
        self.audio_model.patch_embed = PatchEmbed_new(
            img_size=(512, 128), 
            patch_size=(16, 16), 
            in_chans=1,
            embed_dim=768, 
            stride=16
        )
        self.audio_model.pos_embed = nn.Parameter(emb, requires_grad=False)
        msg = self.audio_model.load_state_dict(ckpt, strict=False)
        print('Audio checkpoint loading: ', msg)

    def forward(self, audio):
        """仅接收音频输入，返回回归结果"""
        # 音频输入形状: [batch_size, t=16, ...]（t=16为时序长度）
        n, t, *_ = audio.shape
        assert t == 16, f"音频时序长度必须为16，实际为{t}"
        B = n  # 批次大小

        try:
            # 仅处理音频特征（无图像交互）
            for ii in range(len(self.audio_model.blocks)):
                audio = self.audio_model.forward_block_pre(ii, audio)
                # 音频独立更新（无图像特征参与）
                audio = self.audio_model.forward_block_post(ii, audio, None)

            # 调整音频特征形状以匹配时序网络
            # 音频特征形状: [n, t=16, 768]
            audio_proj = self.audio_proj(audio)  # 投影到512维

            # 输入时序网络
            video_features = self.temporal_net(audio_proj)

            # 回归输出
            output = self.our_classifier(video_features)
            return output

        except Exception as e:
            print(f"处理t={t}时发生异常: {e}", flush=True)
            return torch.zeros((n, 1), dtype=torch.float32, device=audio.device)