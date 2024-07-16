import os
import cv2
import numpy as np
import torch
import lpips
import scipy
import skimage
from skimage.metrics import structural_similarity as compare_ssim

# def rgb_evaluation(gts, predicts, savedir):
#     assert gts.max() <= 1
#     gts = gts.astype(np.float32)
#     predicts = predicts.astype(np.float32)
#     ssim_list = []
#     lpips_list = []
#     mse = ((gts - predicts)**2).mean(-1).mean(-1).mean(-1)
#     print(mse.shape)
#     psnr = (-10*np.log10(mse)).mean()
#     lpips_metric = lpips.LPIPS(net='alex', version='0.1')
#     gts_torch = torch.from_numpy((2*gts - 1).transpose(0, 3, 1, 2)).type(torch.FloatTensor).cuda()
#     predicts_torch = torch.from_numpy((2*predicts - 1).transpose(0, 3, 1, 2)).type(torch.FloatTensor).cuda()

#     for i in range(int(np.ceil(gts_torch.shape[0] / 10.0))):
#         temp = lpips_metric(gts_torch[i*10:(i + 1)*10], predicts_torch[i*10:(i + 1)*10])
#         lpips_list.append(temp.cpu().numpy())
#     lpips_ = np.concatenate(lpips_list, 0).mean()

#     for i in range(gts.shape[0]):
#         gt = gts[i]
#         predict = predicts[i]
#         # ssim_list.append(skimage.measure.compare_ssim(gt, predict, multichannel=True))
#         ssim_list.append(compare_ssim(gt, predict, win_size=3, multichannel=True, data_range=(0, 1)))

#     ssim = np.array(ssim_list).mean()

#     with open(os.path.join(savedir, 'rgb_evaluation.txt'), 'w') as f:
#         result = 'psnr: {0}, ssim: {1}, lpips: {2}'.format(psnr, ssim, lpips_)
#         f.writelines(result)
#         print(result)




''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

__LPIPS__ = {}
def init_lpips(net_name):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to('cuda')

def rgb_lpips(np_gt, np_im, net_name):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to('cuda')
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to('cuda')
    return __LPIPS__[net_name](gt, im, normalize=True).item()



def rgb_evaluation(gts, predicts):
    gts = gts.astype(np.float32)
    predicts = predicts.astype(np.float32)
    mse = ((gts - predicts)**2).mean(-1).mean(-1).mean(-1)
    psnr = (-10*np.log10(mse)).mean()

    ssims,l_alex,l_vgg=[],[],[]
    for i in range(gts.shape[0]):
        gt = gts[i]
        predict = predicts[i]
        ssim = rgb_ssim(predict, gt, 1)
        ssims.append(ssim)

        # l_a = rgb_lpips(gt, predict, 'alex')
        l_v = rgb_lpips(gt, predict, 'vgg')
        # l_alex.append(l_a)
        l_vgg.append(l_v)

    ssim = np.mean(np.asarray(ssims))
    # l_a = np.mean(np.asarray(l_alex))
    l_v = np.mean(np.asarray(l_vgg))

    # with open(os.path.join(savedir, 'rgb_evaluation.txt'), 'w') as f:
    #     result = 'psnr: {0}, ssim: {1}, lpips: {2}'.format(psnr, ssim, l_a)
    #     f.writelines(result)
    #     print(result)

    return psnr, ssim, l_v

