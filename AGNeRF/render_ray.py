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


import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    '''

    # M = weights.shape[1]
    # weights += 1e-5
    # # Get pdf
    # pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [N_rays, M]
    # cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    # cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) # [N_rays, M+1]

    # # Take uniform samples
    # if det:
    #     u = torch.linspace(0., 1., N_samples, device=bins.device)
    #     u = u.unsqueeze(0).repeat(bins.shape[0], 1)       # [N_rays, N_samples]
    # else:
    #     u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # # Invert CDF
    # above_inds = torch.zeros_like(u, dtype=torch.long)       # [N_rays, N_samples]
    # for i in range(M):
    #     above_inds += (u >= cdf[:, i:i+1]).long()

    # # random sample inside each bin
    # below_inds = torch.clamp(above_inds-1, min=0)
    # inds_g = torch.stack((below_inds, above_inds), dim=2)     # [N_rays, N_samples, 2]

    # cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    # cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    # bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # # fix numeric issue
    # denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]      # [N_rays, N_samples]
    # denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # t = (u - cdf_g[:, :, 0]) / denom

    # samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1]-bins_g[:, :, 0])
 

 
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    # if pytest:
    #     np.random.seed(0)
    #     new_shape = list(cdf.shape[:-1]) + [N_samples]
    #     if det:
    #         u = np.linspace(0., 1., N_samples)
    #         u = np.broadcast_to(u, new_shape)
    #     else:
    #         u = np.random.rand(*new_shape)
    #     u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous().to('cuda') 
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

### 改为citynerf
def sample_along_camera_ray(rays_o, rays_d, scene_scaling_factor,scene_origin,
                            N_samples,
                            det=False):
    '''
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    '''
    N_rays = rays_o.shape[0]
    globe_center = torch.tensor(scene_origin * scene_scaling_factor).float()
        
    # 6371011 is earth radius, 250 is the assumed height limitation of buildings in the scene
    earth_radius = 6371011 * scene_scaling_factor
    earth_radius_plus_bldg = (6371011+250) * scene_scaling_factor # 假设建筑物最高250米
    
    ## intersect with building upper limit sphere
    # 与建筑上限球相交
    # 这里的rays_start具体是多少？
    delta = (2*torch.sum((rays_o-globe_center) * rays_d, dim=-1))**2 - 4*torch.norm(rays_d, dim=-1)**2 * (torch.norm((rays_o-globe_center), dim=-1)**2 - (earth_radius_plus_bldg)**2)
    d_near = (-2*torch.sum((rays_o-globe_center) * rays_d, dim=-1) - delta**0.5) / (2*torch.norm(rays_d, dim=-1)**2)
    rays_start = rays_o + (d_near[...,None]*rays_d)
    
    ## intersect with earth
    # 与地球相交
    # 这里的rays_end具体是多少？
    delta = (2*torch.sum((rays_o-globe_center) * rays_d, dim=-1))**2 - 4*torch.norm(rays_d, dim=-1)**2 * (torch.norm((rays_o-globe_center), dim=-1)**2 - (earth_radius)**2)
    d_far = (-2*torch.sum((rays_o-globe_center) * rays_d, dim=-1) - delta**0.5) / (2*torch.norm(rays_d, dim=-1)**2)
    rays_end = rays_o + (d_far[...,None]*rays_d)

    ## compute near and far for each ray
    # near具体是多少？第一个batch中 最大36.45，最小3.28
    # far具体是多少？第一个batch中 最大48.1078，最小6.0113，
    # 具体代表什么含义？怎么和NerfingMVS的深度结合？
    new_near = torch.norm(rays_o - rays_start, dim=-1, keepdim=True)
    near = new_near * 0.9 # (2048,1) 每条光线都有一个near一个far 而且还不一样?
    
    new_far = torch.norm(rays_o - rays_end, dim=-1, keepdim=True)
    far = new_far * 1.1 # (2048,1) 
    
    # disparity sampling for the first half and linear sampling for the rest
    t_vals_lindisp = torch.linspace(0., 1., steps=N_samples).to('cuda') 
    z_vals_lindisp = 1./(1./near * (1.-t_vals_lindisp) + 1./far * (t_vals_lindisp))
    z_vals_lindisp_half = z_vals_lindisp[:,:int(N_samples*2/3)]

    linear_start = z_vals_lindisp_half[:,-1:]
    t_vals_linear = torch.linspace(0., 1., steps=N_samples-int(N_samples*2/3)+1).to('cuda')
    z_vals_linear_half = linear_start * (1-t_vals_linear) + far * t_vals_linear
    
    z_vals = torch.cat((z_vals_lindisp_half, z_vals_linear_half[:,1:]), -1)
    z_vals, _ = torch.sort(z_vals, -1)
    z_vals = z_vals.expand([N_rays, N_samples])

    # if perturb > 0.:
    
    if not det:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape).to('cuda')
        z_vals = lower + (upper - lower) * t_rand

    # means, cov_diags = cast(rays_o, rays_d, radii, z_vals)
    # raw = network_query_fn(means, cov_diags, rays_d, network_fn)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # pts [N_rays, N_samples, 3]
    return pts, z_vals

########################################################################################################################
# ray rendering of nerf
########################################################################################################################

# def raw2outputs(raw, z_vals, mask, white_bkgd=False):
#     '''
#     :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
#     :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
#     :param ray_d: [N_rays, 3]
#     :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
#     '''
#     rgb = raw[:, :, :3]     # [N_rays, N_samples, 3]
#     sigma = raw[:, :, 3]    # [N_rays, N_samples]

#     # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
#     # very different scales, and using interval can affect the model's generalization ability.
#     # Therefore we don't use the intervals for both training and evaluation.
#     sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

#     # point samples are ordered with increasing depth
#     # interval between samples
#     dists = z_vals[:, 1:] - z_vals[:, :-1]
#     dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

#     alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

#     # Eq. (3): T
#     T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
#     T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

#     # maths show weights, and summation of weights along a ray, are always inside [0, 1]
#     weights = alpha * T     # [N_rays, N_samples]
#     rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

#     if white_bkgd:
#         rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

#     mask = mask.float().sum(dim=1) > 8  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
#     depth_map = torch.sum(weights * z_vals, dim=-1)     # [N_rays,]

#     # ret = OrderedDict([('rgb', rgb_map),
#     #                    ('depth', depth_map),
#     #                    ('weights', weights),                # used for importance sampling of fine samples
#     #                    ('mask', mask),
#     #                    ('alpha', alpha),
#     #                    ('z_vals', z_vals)
#     #                    ])
#     ret = OrderedDict([('rgb', rgb_map),
#                        ('depth', depth_map),
#                        ('weights', weights),                # used for importance sampling of fine samples
#                        ('alpha', alpha),
#                        ('z_vals', z_vals)
#                        ])

#     return ret


def raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd=False):
    '''
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    '''
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).cuda()], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape).cuda() * torch.tensor(raw_noise_std).cuda()
        
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map).cuda(), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    ret = OrderedDict([('rgb', rgb_map),
                       ('depth', depth_map),
                       ('weights', weights),                # used for importance sampling of fine samples
                       ('alpha', alpha),
                       ('z_vals', z_vals)
                       ])

    return ret


def render_rays(args,
                ray_batch,
                model,
                featmaps,
                projector,
                N_samples,
                inv_uniform=False,
                N_importance=0,
                det=False,
                white_bkgd=False):
    '''
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    '''

    ret = {'outputs_coarse': None,
           'outputs_fine': None}
    ray_o, ray_d = ray_batch["ray_o"], ray_batch["ray_d"] # 'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 

    # pts: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]
    pts, z_vals = sample_along_camera_ray(rays_o=ray_o,
                                          rays_d=ray_d,
                                          scene_scaling_factor=ray_batch['scene_scaling_factor'],
                                          scene_origin=ray_batch['scene_origin'],
                                          N_samples=N_samples, det=det) # 每条光线上采样点
    N_rays, N_samples = pts.shape[:2]
    # ray_batch['camera'].shape = [1, 34]   featmaps[0] [num_相似pose, coarse_feat_dim, H//4, W//4]
    rgb_feat, ray_diff, mask = projector.compute(pts, ray_batch['camera'],
                                                 ray_batch['src_rgbs'],
                                                 ray_batch['src_cameras'],
                                                 featmaps=featmaps[0])  # [N_rays, N_samples, N_views, x] 每条光线的采样点投影到相邻的featuremap上然后插值获得特征



    # pixel_mask = mask[..., 0].sum(dim=2) > 1   # [N_rays, N_samples], should at least have 2 observations
    
    # mine
    # plane a 
    # PE
    # embedded = embed_fn(pts)  # 对输入进行位置编码，得到编码后的结果，是一个 array 数组
    # input_dirs = ray_d[:,None].expand(pts.shape)
    # embedded_dirs = embeddirs_fn(input_dirs)
    # raw_coarse = model.net_coarse(embedded, embedded_dirs, rgb_feat, ray_diff, mask)   # [N_rays, N_samples, 4]
    
    # plane b
    raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask, pts, ray_d)   # [N_rays, N_samples, 4]
  
    
    # offical
    # raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask)   # [N_rays, N_samples, 4]

    # outputs_coarse = raw2outputs(raw_coarse, z_vals, pixel_mask,
    #                              white_bkgd=white_bkgd)

    # mine
    outputs_coarse = raw2outputs(raw_coarse, z_vals, ray_d, args.raw_noise_std, white_bkgd=white_bkgd)



    ret['outputs_coarse'] = outputs_coarse

    if N_importance > 0:
        assert model.net_fine is not None
        # detach since we would like to decouple the coarse and fine networks
        weights = outputs_coarse['weights'].clone().detach()            # [N_rays, N_samples]

        if inv_uniform: # 这个应该不要！！！！
            inv_z_vals = 1. / z_vals
            inv_z_vals_mid = .5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])   # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
            inv_z_vals = sample_pdf(bins=torch.flip(inv_z_vals_mid, dims=[1]),
                                    weights=torch.flip(weights, dims=[1]),
                                    N_samples=N_importance, det=det)  # [N_rays, N_importance]
            z_samples = 1. / inv_z_vals
        else:
            # take mid-points of depth samples
            z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])   # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
            z_samples = sample_pdf(bins=z_vals_mid, weights=weights,
                                   N_samples=N_importance, det=det)  # [N_rays, N_importance]


        z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

        # samples are sorted with increasing depth
        z_vals, _ = torch.sort(z_vals, dim=-1)

        # offical
        # N_total_samples = N_samples + N_importance
        # viewdirs = ray_batch['ray_d'].unsqueeze(1).repeat(1, N_total_samples, 1)
        # ray_o = ray_batch['ray_o'].unsqueeze(1).repeat(1, N_total_samples, 1)
        # pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]



        # mine
        pts = ray_o[...,None,:] + ray_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        rgb_feat_sampled, ray_diff, mask = projector.compute(pts, ray_batch['camera'],
                                                             ray_batch['src_rgbs'],
                                                             ray_batch['src_cameras'],
                                                             featmaps=featmaps[1])
        # pixel_mask = mask[..., 0].sum(dim=2) > 1  # [N_rays, N_samples]. should at least have 2 observations
        
  
    
        # mine
        # plane a 
        # PE
        # embedded = embed_fn(pts)  # 对输入进行位置编码，得到编码后的结果，是一个 array 数组
        # input_dirs = ray_d[:,None].expand(pts.shape)
        # embedded_dirs = embeddirs_fn(input_dirs)
        # raw_fine = model.net_fine(embedded, embedded_dirs, rgb_feat_sampled, ray_diff, mask)
        
        # plane b
        raw_fine = model.net_fine(rgb_feat_sampled, ray_diff, mask, pts, ray_d)   # [N_rays, N_samples, 4]

    
        # offical
        # raw_fine = model.net_fine(rgb_feat_sampled, ray_diff, mask)

        # outputs_fine = raw2outputs(raw_fine, z_vals, pixel_mask,
        #                            white_bkgd=white_bkgd)

        # mine
        outputs_fine = raw2outputs(raw_fine, z_vals, ray_d, args.raw_noise_std, white_bkgd=white_bkgd)


        ret['outputs_fine'] = outputs_fine

    return ret
