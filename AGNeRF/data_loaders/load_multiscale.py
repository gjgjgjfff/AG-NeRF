import numpy as np
import os
import json
import cv2
import imageio
import torch


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_google_data(basedir, factor=None):
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')][0]
    sh = np.array(cv2.imread(img0).shape)


    sfx = ''

    if factor is not None and factor != 1:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')]
    

    data = json.load(open(os.path.join(basedir, 'poses_enu.json')))
    poses = np.array(data['poses'])[:, :-2].reshape([-1, 3, 5])
    poses[:, :2, 4] = np.array(sh[:2]//factor).reshape([1, 2])
    poses[:, 2, 4] = poses[:,2, 4] * 1./factor 
    poses = poses.astype(np.float32)

    scene_scaling_factor = np.float32(data['scene_scale'])
    scene_origin = np.array(data['scene_origin'])
    scene_origin = scene_origin.astype(np.float32)
    # scale_split = data['scale_split']


    return imgfiles, poses, scene_scaling_factor, scene_origin



def load_multiscale_data(basedir, factor=4):
    imgfiles, poses, scene_scaling_factor, scene_origin = _load_google_data(basedir, factor=factor)
    return imgfiles, poses, scene_scaling_factor, scene_origin
