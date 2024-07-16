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

import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../')
from .data_utils import random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses
from .load_multiscale import load_multiscale_data

import cv2




def read_img(rgb_file, factor):
    sh = np.array(cv2.imread(rgb_file).shape)
    im = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
    if im.shape[-1] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    im = cv2.resize(im, (sh[1]//factor, sh[0]//factor), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32) / 255
    return im


class Citynerf(Dataset):
    def __init__(self, args, mode, **kwargs):
        base_dir = os.path.join(args.datadir)
        self.args = args
        self.factor = args.factor
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        
        self.test_rgb_files = []
        
        self.scene_scaling_factor = 0
        self.scene_origin = 0
        # scenes = os.listdir(base_dir)
                
        if mode == 'train':
            scenes = args.train_scenes
        else:
            scenes = args.eval_scenes

        for i, scene in enumerate(scenes):
            scene_path = os.path.join(base_dir, scene)
            rgb_files, poses, scene_scaling_factor, scene_origin = load_multiscale_data(scene_path, args.factor)
            
            # images = images[scale_split[args.cur_stage]:]
            # img_names = img_names[scale_split[args.cur_stage]:]
            # poses = poses[scale_split[args.cur_stage]:]

            print('num_image:', poses.shape[0])

            if args.holdout > 0:
                print('Auto holdout,', args.holdout)
                i_test = np.arange(poses.shape[0])[::args.holdout]

            i_train = np.array([i for i in np.arange(int(poses.shape[0])) if
                            (i not in i_test)])

            self.scene_scaling_factor = scene_scaling_factor
            self.scene_origin = scene_origin

            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            self.test_rgb_files.append(np.array(rgb_files)[i_test].tolist())


            if mode == 'train':
                i_render = i_train
                print('num_i_train:',len(i_train), '\n')
                for file in self.train_rgb_files:
                    print(file,'\n')
            else:
                i_render = i_test
                print('num_i_test:',len(i_test), '\n')
                for file in self.test_rgb_files:
                    print(file,'\n')


            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_train_set_ids.extend([i]*num_render)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.

        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            id_render = train_rgb_files.index(rgb_file)
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=3)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                min(self.num_source_views*subsample_factor, 20),
                                                tar_id=id_render,
                                                angular_dist_method='dist')

        # # 根据掩码筛选符合条件的 pose IDs
        # nearest_pose_ids = nearest_pose_ids[mask]
        assert nearest_pose_ids is not None
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)
        # print('id_render:', id_render, '\n', 'nearest_pose_ids:', nearest_pose_ids)
        assert id_render not in nearest_pose_ids

        # occasionally include input image
        # if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
        # nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            # src_rgb = read_img(train_rgb_files[id], self.factor)
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                              train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == 'train':
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w))

        if self.mode == 'train' and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)


        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'scene_scaling_factor': self.scene_scaling_factor,
                'scene_origin': self.scene_origin,
                }

