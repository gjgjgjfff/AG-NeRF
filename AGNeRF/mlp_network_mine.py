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
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

from AGNeRF.Position_Encoding import Embedder


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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


class Attention(nn.Module):              
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x, k):
        b1, b2, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)           # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b c n (h d) -> b c h n d', h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b c h i d, b c h j d -> b c h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b c h i j, b c h j d -> b c h i d', attn, v)
        out = rearrange(out, 'b c h n d -> b c n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate, heads, dim_head):
        super().__init__()
            
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dp_rate)

    def forward(self, x, k):
        x = x.unsqueeze(2)
        x = x.expand(-1, -1, k.shape[-2], -1)
        residue = x
        x = self.attn_norm(x)
        x = self.attn(x, k)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue
            
        return x





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

        for _ in range(self.transformerdepth):
            # view transformer
            view_trans = Transformer(dim=self.net_width, ff_hid_dim=int(self.net_width * 4), ff_dp_rate=0.1, attn_dp_rate=0.1, heads=2, dim_head=32)
            self.view_crosstrans.append(view_trans)

                
    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d): # pts [N_rays, N_samples, 3]]  ray_d [N_rays, 3]
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = viewdirs[:,None].expand(pts.shape)
        input_views = self.view_enc(viewdirs)
        input_pts = self.pos_enc(pts)

        rgb_feat = self.rgbfeat_fc(rgb_feat)
        q = rgb_feat.max(dim=2)[0] 

        for i, crosstrans in enumerate(self.view_crosstrans):
            # view transformer to update q
            q = crosstrans(q, rgb_feat) 
        
        # q = q.mean(dim=2)
        q = rgb_feat.max(dim=2)[0]
        
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