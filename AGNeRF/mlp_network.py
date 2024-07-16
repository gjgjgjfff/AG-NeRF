# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

from AGNeRF.Position_Encoding import Embedder


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q) # [n_ray, n_sample, dim]
        # q = q.unsqueeze(2).repeat(1, 1, k.shape[-2], 1)
        k = self.k_fc(k)  # [n_ray, n_sample, V, dim]
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        # attn = torch.matmul(k.transpose(-2, -1), q)
        
        attn = k - q[:, :, None, :] + pos # [n_ray, n_sample, n_view, dim]
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x



# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# class DeformableTransformerEncoderLayer(nn.Module):
#     def __init__(self,
#                  d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=1, n_points=4):
#         super().__init__()

#         # self attention
#         self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, src):
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
#         src = src + self.dropout3(src2)
#         src = self.norm2(src)
#         return src

#     def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
#         # self attention
#         src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         # ffn
#         src = self.forward_ffn(src)

#         return src









class IBRNet(nn.Module):
    def __init__(self, args, net_depth=8, net_width=64, input_ch_feat=32, skips=[4]):
        """ 
        """
        super(IBRNet, self).__init__()
        self.net_depth = net_depth
        self.net_width = net_width
        self.input_ch_feat = input_ch_feat
        self.skips = args.skips
        self.transformerdepth = args.transformerdepth


        ### encoding
        self.pos_enc = Embedder(
                    input_dims=3,
                    include_input=True,
                    max_freq_log2=args.multires - 1,
                    num_freqs=args.multires,
                    log_sampling=True,
                    periodic_fns=[torch.sin, torch.cos],
                )
        self.input_pts_ch = self.pos_enc.out_dim

        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=args.multires_views - 1,
            num_freqs=args.multires_views,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

        self.input_views_ch = self.view_enc.out_dim

        ### linear
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_pts_ch + self.net_width, self.net_width)] + [nn.Linear(self.net_width, self.net_width) if i not in self.skips else nn.Linear(self.net_width + self.input_pts_ch, self.net_width) for i in range(self.net_depth-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(self.input_views_ch + self.net_width, self.net_width//2)])

        self.feature_linear = nn.Linear(self.net_width, self.net_width)
        self.alpha_linear = nn.Linear(self.net_width, 1)
        self.rgb_linear = nn.Linear(self.net_width//2, 3)


        ### GNT

        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(self.input_ch_feat + 3, self.net_width),
            nn.ReLU(),
            nn.Linear(self.net_width, self.net_width),
        )

        self.view_crosstrans = nn.ModuleList([])

        # for _ in range(self.transformerdepth):
        #     # view transformer
        #     view_trans = Transformer2D(
        #         dim=self.net_width,
        #         ff_hid_dim=int(self.net_width * 4),
        #         ff_dp_rate=0.1,
        #         attn_dp_rate=0.1,
        #     )
        #     self.view_crosstrans.append(view_trans)

            

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d): # pts [N_rays, N_samples, 3]]  ray_d [N_rays, 3]
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = viewdirs[:,None].expand(pts.shape)
        input_views = self.view_enc(viewdirs)
        input_pts = self.pos_enc(pts)

        rgb_feat = self.rgbfeat_fc(rgb_feat)
        q = rgb_feat.max(dim=2)[0] # maxpool 取了所有source_view中个里面的最大的？  为啥特征要是最大的？ [batch_size,num_point,d]
        # q = rgb_feat.mean(dim=2) # [batch_size,num_point,d]

        # for i, crosstrans in enumerate(self.view_crosstrans):
        #     # view transformer to update q
        #     q = crosstrans(q, rgb_feat, ray_diff, mask) 

        h = torch.cat([input_pts, q], dim=-1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h0 = torch.cat([feature, input_views], -1)
    
        for i, l in enumerate(self.views_linears):
            h0 = self.views_linears[i](h0)
            h0 = F.relu(h0)

        rgb = self.rgb_linear(h0)
        outputs = torch.cat([rgb, alpha], -1)



        return outputs