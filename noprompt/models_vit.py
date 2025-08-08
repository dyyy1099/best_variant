# --------------------------------------------------------
# References:
# timm: https://github.com/huggingface/pytorch-image-models
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

"""可学习与模态融合模块（已移除可学习提示）"""
from functools import partial

import torch
from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn

class VisionTransformer2(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, n_seq=196, n_progr=0, n_frames=16, **kwargs):
        # 将n_progr默认值设为0，实现无提示效果
        super(VisionTransformer2, self).__init__(** kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

        self.n_seq = n_seq
        self.n_progr = n_progr  # 可学习提示的数量，0表示无提示
        self.n_frames = n_frames

        self.latent_dim = 128
       
        # 仅当n_progr > 0时创建提示参数
        if self.n_progr > 0:
            self.learnable_prompts_init = nn.Parameter(
                torch.randn(self.n_progr * (len(self.blocks)//6), 768) * (768 **-0.5)
            )
            self.learnable_prompts_progr = nn.ParameterList([
                nn.Parameter(torch.randn(self.n_progr, 768) * (768** -0.5)) 
                for _ in range(len(self.blocks)//6)
            ])
        else:
            # 提示数量为0时不创建参数
            self.learnable_prompts_init = None
            self.learnable_prompts_progr = None

        self.audio_proj_pre = nn.ModuleList([
            nn.Sequential(nn.Linear(768, self.latent_dim), nn.LayerNorm(self.latent_dim)) 
            for _ in range(len(self.blocks))
        ])
        self.temporal_pre = nn.ModuleList([
            nn.Linear(768, self.latent_dim) for _ in range(len(self.blocks))
        ])

        self.temporal_pre_norm = nn.ModuleList([
            nn.LayerNorm(self.latent_dim) for _ in range(len(self.blocks))
        ])
        self.temporal_att_post = nn.ModuleList([
            nn.Sequential(nn.Linear(self.latent_dim, 768), nn.GELU()) 
            for _ in range(len(self.blocks))
        ])
        self.all_gate = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(len(self.blocks))
        ])
   
    def forward_block_pre(self, ii, x, B):
        if ii == 0:
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            
            # 仅当n_progr > 0时拼接提示向量
            if self.n_progr > 0 and self.learnable_prompts_init is not None:
                x = torch.cat((cls_tokens, x, self.learnable_prompts_init.expand(B, -1, -1)), dim=1)
            else:
                x = torch.cat((cls_tokens, x), dim=1)  # 无提示时仅拼接cls和特征
            
            x = x + self.pos_embed
            x = self.pos_drop(x)

        x = self.blocks[ii](x)
        return x

    def forward_block_post(self, ii, x, x_t, B):
        x_t = self.temporal_att_post[ii](x_t)
        x = x + nn.functional.tanh(self.all_gate[ii]) * x_t.unsqueeze(2).view(B, -1, 768)

        # 仅当n_progr > 0时更新提示向量
        if self.n_progr > 0 and self.learnable_prompts_progr is not None:
            if ii % 6 == 0:
                prompts_progr = self.learnable_prompts_progr[ii//6].expand(B, -1, -1)
                x[:, self.n_seq+1 + ii//6*self.n_progr : self.n_seq+1 + (ii//6+1)*self.n_progr, :] += prompts_progr

        if ii == (len(self.blocks) - 1):
            if self.global_pool:
                x = x[:, 1:, :]  # without cls token
                x = x.mean(dim=1)  # global average pooling
                outcome = self.fc_norm(x)
                return outcome
            else:
                x = self.norm(x)
                outcome = x[:, 0]
                return outcome
        return x
 
 
    def forward_features(self, x, audio=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # 仅当n_progr > 0时拼接提示向量
        if self.n_progr > 0 and self.learnable_prompts_init is not None:
            x = torch.cat((cls_tokens, x, self.learnable_prompts_init.expand(B, -1, -1)), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)  # 无提示时仅拼接cls和特征
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for ii, blk in enumerate(self.blocks): 
            # 仅当n_progr > 0时获取提示向量
            prompts_progr = self.learnable_prompts_progr[ii].expand(B, -1, -1) if \
                (self.n_progr > 0 and self.learnable_prompts_progr is not None) else None

            x = blk(x)

            # 时间特征处理
            x_t = x[:,0,:].contiguous().view(B // 16, 16, x.shape[-1]) 
            x_t = x_t + self.temporal_pos_embed		
            x_t = self.temporal_pre[ii](x_t)

            qs = self.learnable_q[ii].expand(B // 16, -1, -1)
            qs = self.norm_qs[ii](qs)
            x_t_1, _ = self.context_att[ii](qs, x_t, x_t, need_weights=False)

            # 音频特征融合
            if audio is not None and ii < len(audio):
                x_a = self.audio_proj_pre[ii](audio[ii])                        
                x_t_2, _ = self.audio_att[ii](qs, x_a, x_a, need_weights=False)
                x_t = x_t + nn.functional.tanh(self.audio_gate[ii])*x_t_2 + nn.functional.tanh(self.context_gate[ii])*x_t_1

            x_t = self.temporal_att_post[ii](x_t)
            x[:,1:197,:] = x[:,1:197,:] + nn.functional.tanh(self.all_gate[ii]) * x_t.unsqueeze(2).view(B, -1, 768)

            # 仅当n_progr > 0时更新提示向量
            if self.n_progr > 0 and self.learnable_prompts_progr is not None and ii != 11:
                x[:, 197 + ii*self.n_progr : 197 + (ii+1)*self.n_progr, :] += prompts_progr

        if self.global_pool:
            x = x[:, 1:, :]
            x = x.mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            outcome = x[:, 0]

        return outcome

    # borrow from timm
    def forward(self, x, ret_feature=False):
        x = self.forward_features(x)
        feature = x
        if getattr(self, 'head_dist', None) is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        if ret_feature:
            return x, feature
        else:
            return x


# setup model archs
VIT_KWARGS_BASE = dict(
    mlp_ratio=4, 
    qkv_bias=True,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
)

VIT_KWARGS_PRESETS = {
    'tiny': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    'small': dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    'base': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    'large': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    'huge': dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    'giant': dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11),
    'gigantic': dict(patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=64/13),
}

def create_vit_model(preset=None, creator=None, **kwargs):
    preset = 'base' if preset is None else preset.lower()
    all_kwargs = dict()
    all_kwargs.update(VIT_KWARGS_BASE)
    all_kwargs.update(VIT_KWARGS_PRESETS[preset])
    all_kwargs.update(kwargs)
    if creator is None:
        creator = VisionTransformer2
    return creator(** all_kwargs)

# 模型实例化入口
vit_base_patch16 = partial(create_vit_model, preset='base')
    