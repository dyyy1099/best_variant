from torch import nn
from models.Temporal_Model import *
import torchaudio
import math
from AudioMAE import audio_models_vit
from timm.models.layers import to_2tuple
from models import models_vit
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
    # function from timm
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    if gs_old is None:
        gs_old = (int(math.sqrt(len(posemb_grid))), int(math.sqrt(len(posemb_grid))))

    if gs_new is None or not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old[0], gs_old[1], -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


class PatchEmbed_new(nn.Module):
    '''
    copied from AudioMAE
    '''
    """ Flexible Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)  # with overlapped patches
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        # self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        # print('shape: ', x.shape)
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class GenerateModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        self.our_classifier = torch.nn.Linear(512, 1)
        self.vision_proj = torch.nn.Linear(768, 512)

        self.n_image = (args.img_size // 16) ** 2
        self.n_progr = 3

        self._build_image_model(img_size=args.img_size)



    def _build_image_model(self, model_name='vit_base_patch16', ckpt_path='./mae_face_pretrain_vit_base.pth',
                           global_pool=False, num_heads=12, drop_path_rate=0.1, img_size=224, n_frames=16):

        self.image_encoder = getattr(models_vit, model_name)(
            global_pool=global_pool,
            num_classes=num_heads,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
            n_seq=self.n_image,
            n_progr=self.n_progr,
            n_frames=n_frames,
        )

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        orig_pos_embed = checkpoint_model['pos_embed']
        new_posemb = resize_pos_embed(orig_pos_embed, self.image_encoder.pos_embed)  # use PyTorch function linked above
        checkpoint_model['pos_embed'] = new_posemb

        msg = self.image_encoder.load_state_dict(checkpoint_model, strict=False)
        print('Image checkpoint loading: ', msg)
        pos_embed = torch.randn(1, self.image_encoder.pos_embed.size(1) + (
            len(self.image_encoder.blocks)) * self.n_progr // 6, 768)
        pos_embed[:, :-(len(self.image_encoder.blocks)) * self.n_progr // 6, :] = self.image_encoder.pos_embed
        self.image_encoder.pos_embed = nn.Parameter(pos_embed)

    def forward(self, image, audio=None):  # audio参数设为可选，兼容原有调用方式
            # 仅处理图像输入
            n, t, c, h, w = image.shape
            image = image.contiguous().view(-1, c, h, w)  # 展平时间维度

            # 检查输入帧数量
            if t != 16:
                print(f"警告：t的值为{t}（预期16），原始输入形状为{image.shape}", flush=True)

            B = image.shape[0]

            try:
                for ii in range(len(self.image_encoder.blocks)):
                    image = self.image_encoder.forward_block_pre(ii, image, B)
                    
                    image_lowdim_temp = self.image_encoder.temporal_pre[ii](image)
                    image_lowdim_norm = self.image_encoder.temporal_pre_norm[ii](image_lowdim_temp)
                    
                    image = self.image_encoder.forward_block_post(ii, image, image_lowdim_norm, B)

                image = image.contiguous().view(n, t, -1)
                image = self.vision_proj(image)  # 仅使用图像特征

                # 时间融合
                video_features = self.temporal_net(image)

                # 输出预测结果
                output = self.our_classifier(video_features)
                output = torch.sigmoid(output) * 63  
                output = torch.clamp(output, min=0, max=63)  

                return output
            except Exception as e:
                print(f"处理t={t}时发生异常: {e}", flush=True)
                return torch.zeros((n, 1), dtype=torch.int32, device=image.device)
