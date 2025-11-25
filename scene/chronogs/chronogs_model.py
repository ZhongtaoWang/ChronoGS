#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
import torch
import math
import numpy as np
from torch import nn
from einops import repeat
from functools import reduce
from torch_scatter import scatter_max
from utils.general_utils import get_expon_lr_func, knn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from scene.embedding import Embedding
from scene.basic_model import BasicModel
from tqdm import tqdm


import torch
import torch.nn.functional as F

    
class GaussianModel(BasicModel):

    def __init__(self, **model_kwargs):

        for key, value in model_kwargs.items():
            setattr(self, key, value)

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)

        self.opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        if not hasattr(self, 'feat_dim'):
            self.feat_dim = self.varing_feat_dim + self.base_feat_dim
                
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()

    
    def setup_functions(self):
        super().setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(self.view_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim + (0 if hasattr(self, 'use_global_feat_in_opacity') and self.use_global_feat_in_opacity == False else self.global_feat_dim), self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()
        
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim + (0 if hasattr(self, 'use_global_feat_in_cov') and self.use_global_feat_in_cov == False else self.global_feat_dim), self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).cuda()
    
        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim+self.global_feat_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.global_feat_dim > 0:
            self.embedding_global_feat.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.global_feat_dim > 0:
            self.embedding_global_feat.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        param_dict = {}
        param_dict['optimizer'] = self.optimizer.state_dict()
        param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
        param_dict['cov_mlp'] = self.mlp_cov.state_dict()
        param_dict['color_mlp'] = self.mlp_color.state_dict()
        if self.use_feat_bank:
            param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
        if self.global_feat_dim > 0:
            param_dict['global_feat'] = self.embedding_global_feat.state_dict()
        return (
            self._anchor,
            self._offset,
            self._scaling,
            self._rotation,
            self.opacity_accum, 
            self.anchor_demon,
            self.offset_gradient_accum,
            self.offset_denom,
            param_dict,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self._anchor,
        self._offset,
        self._scaling,
        self._rotation,
        self.opacity_accum, 
        self.anchor_demon,
        self.offset_gradient_accum,
        self.offset_denom,
        param_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(param_dict['optimizer'])
        self.mlp_opacity.load_state_dict(param_dict['opacity_mlp'])
        self.mlp_cov.load_state_dict(param_dict['cov_mlp'])
        self.mlp_color.load_state_dict(param_dict['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(param_dict['feature_bank_mlp'])
        if self.global_feat_dim > 0:
            self.embedding_global_feat.load_state_dict(param_dict['global_feat'])

    @property
    def get_anchor(self):
        return self._anchor
        
    @property
    def get_anchor_feat(self):
        return self._anchor_feat

    @property
    def get_offset(self):
        return self._offset

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def set_global_feat(self, num_cameras):
        if self.global_feat_dim > 0:
            print(f"global_feat_dim: {self.global_feat_dim}")
            self.embedding_global_feat = Embedding(num_cameras, self.global_feat_dim).cuda()
        else:
            self.embedding_global_feat = None
            
    @property
    def get_global_feat(self):
        return self.embedding_global_feat
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    def get_voxel_id_for_point(self, point_idx):
        """
        获取指定点对应的体素ID
        
        Args:
            point_idx: 原始点的索引
            
        Returns:
            voxel_id: 对应的体素ID
        """
        if hasattr(self, 'point_to_voxel_map'):
            return self.point_to_voxel_map[point_idx]
        else:
            raise ValueError("Point to voxel mapping not available")
    
    def get_points_in_voxel(self, voxel_id):
        """
        获取指定体素内的所有点索引
        
        Args:
            voxel_id: 体素ID
            
        Returns:
            point_indices: 该体素内的所有点索引
        """
        if hasattr(self, 'point_to_voxel_map'):
            return torch.where(self.point_to_voxel_map == voxel_id)[0]
        else:
            raise ValueError("Point to voxel mapping not available")
    
    def get_voxel_statistics(self):
        """
        获取体素化统计信息
        
        Returns:
            dict: 包含体素化统计信息的字典
        """
        if hasattr(self, 'point_to_voxel_map'):
            unique_voxels, counts = torch.unique(self.point_to_voxel_map, return_counts=True)
            return {
                'total_points': self.point_to_voxel_map.shape[0],
                'total_voxels': unique_voxels.shape[0],
                'points_per_voxel_mean': counts.float().mean().item(),
                'points_per_voxel_std': counts.float().std().item(),
                'points_per_voxel_min': counts.min().item(),
                'points_per_voxel_max': counts.max().item(),
            }
        else:
            raise ValueError("Point to voxel mapping not available")

    def voxelize_sample(self, data=None, voxel_size=0.001):
        """
        体素化点云数据，并返回体素中心位置和点到体素的映射关系
        
        Args:
            data: 输入点云数据 [N, 3]
            voxel_size: 体素大小
            
        Returns:
            voxel_centers: 体素中心位置 [M, 3]
            point_to_voxel_map: 每个原始点对应的体素ID [N]
        """
        # 步骤1: 将点云量化到体素网格
        candidate_points = torch.round(data/voxel_size)
        
        # 步骤2: 找到唯一的体素位置，并获取映射关系
        candidates_unique, inverse_indices = torch.unique(candidate_points, return_inverse=True, dim=0)
        
        # 步骤3: 计算体素中心位置
        voxel_centers = (candidates_unique + self.padding) * voxel_size
        
        return voxel_centers, inverse_indices

    def create_from_pcd(self, pcd, spatial_lr_scale, logger,  *args , times = None,):
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(pcd.points).float().cuda()
        if self.voxel_size <= 0:
            init_dist = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            torch.cuda.empty_cache()
                        
        fused_point_cloud, point_to_voxel_map = self.voxelize_sample(points, voxel_size=self.voxel_size)

        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        
        if times is not None:
            self.times = times
            print(f"use times: {times}")
            print(f"feat_dim: {self.feat_dim}")
            anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.base_feat_dim + self.varing_feat_dim * (times))).float().cuda()
        else:
            print(f"use times: None")
            print(f"feat_dim: {self.feat_dim}")
            anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        logger.info(f'Initial Voxel Number: {fused_point_cloud.shape[0]}')
        logger.info(f'Voxel Size: {self.voxel_size}')

        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        logger.info(f'Point to voxel mapping saved: {point_to_voxel_map.shape[0]} points -> {fused_point_cloud.shape[0]} voxels')

    def training_setup(self, training_args):
        
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

        ]
        if self.global_feat_dim > 0:
            l.append({'params': self.embedding_global_feat.parameters(), 'lr': training_args.global_feat_lr_init, "name": "embedding_global_feat"})
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)

        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.global_feat_dim > 0:
            self.global_feat_scheduler_args = get_expon_lr_func(lr_init=training_args.global_feat_lr_init,
                                                        lr_final=training_args.global_feat_lr_final,
                                                        lr_delay_mult=training_args.global_feat_lr_delay_mult,
                                                        max_steps=training_args.global_feat_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.global_feat_dim > 0 and param_group["name"] == "embedding_global_feat":
                lr = self.global_feat_scheduler_args(iteration)
                param_group['lr'] = lr
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        anchor = self._anchor.detach().cpu().numpy()
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, offset, anchor_feat, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], -1, self.n_offsets))

        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    
    def set_times(self, times):
        self.times = times

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def anchor_growing(self, grads, threshold, offset_mask, overlap):

        init_length = self.get_anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self.get_offset * self.get_scaling[:,:3].unsqueeze(dim=1) # [N,K,3]
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size - self.padding).int()
            selected_xyz = all_xyz.reshape([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size - self.padding).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            if overlap:
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.padding * cur_size
            elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size + self.padding * cur_size
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')

            if candidate_anchor.shape[0] > 0:
                new_scaling_shape = list(self.get_scaling.shape)
                new_scaling_shape[0] = candidate_anchor.shape[0]
                new_scaling = torch.ones(new_scaling_shape, dtype=torch.float, device='cuda')*cur_size # *0.05
                new_scaling = torch.log(new_scaling)

                new_offset_shape = list(self.get_offset.shape)
                new_offset_shape[0] = candidate_anchor.shape[0]
                new_offsets = torch.zeros(new_offset_shape, dtype=torch.float, device='cuda')

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).reshape([-1, self._anchor_feat.shape[1]])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                }

                
                
                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([candidate_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
    
    def run_densify(self, iteration, opt):
        # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, opt.densify_grad_threshold, offset_mask, opt.overlap)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # prune anchors
        prune_mask = (self.opacity_accum < opt.min_opacity * self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > opt.update_interval * opt.success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.reshape([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.reshape([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.reshape([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.reshape([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
    def save_mlp_checkpoints(self, path):#split or unite
        mkdir_p(os.path.dirname(path))
        self.eval()
        opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim + (0 if self.use_global_feat_in_opacity == False else self.global_feat_dim)).cuda()))
        opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
        cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim + (0 if self.use_global_feat_in_cov == False else self.global_feat_dim)).cuda()))
        cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
        color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.global_feat_dim).cuda()))
        color_mlp.save(os.path.join(path, 'color_mlp.pt'))
        if self.use_feat_bank:
            feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, self.view_dim).cuda()))
            feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
        if self.global_feat_dim > 0:
            emd = torch.jit.trace(self.embedding_global_feat, (torch.zeros((1,), dtype=torch.long).cuda()))
            emd.save(os.path.join(path, 'embedding_global_feat.pt'))
        self.train()

    def load_mlp_checkpoints(self, path):
        self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
        if self.use_feat_bank:
            self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
        if self.global_feat_dim > 0:
            self.embedding_global_feat = torch.jit.load(os.path.join(path, 'embedding_global_feat.pt')).cuda()
    
    def encode_timestamp(self, timestamp, num_times):

        if isinstance(timestamp, (int, float)) and timestamp == int(timestamp):
            # 整数时间戳，使用onehot编码
            encoded = torch.zeros(num_times, device=self.get_anchor.device)
            encoded[int(timestamp)] = 1.0
            return encoded
        else:
            # 小数时间戳，使用线性插值
            timestamp = float(timestamp)
            t_floor = int(timestamp)
            t_ceil = t_floor + 1
            weight = timestamp - t_floor
            
            encoded = torch.zeros(num_times, device=self.get_anchor.device)
            
            # 确保索引在有效范围内
            if 0 <= t_floor < num_times:
                encoded[t_floor] = 1.0 - weight
            if 0 <= t_ceil < num_times:
                encoded[t_ceil] = weight
                
            return encoded

    def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, ape_code=-1):    

        if hasattr(viewpoint_camera, 'geo_time_stamp'):
            geo_time_stamp = viewpoint_camera.geo_time_stamp
        else:
            geo_time_stamp = viewpoint_camera.time_stamp

        anchor_geo_time_stamps_onehot = torch.zeros(self.get_anchor.shape[0], self.times, device=self.get_anchor.device)
        anchor_time_stamps_onehot = torch.zeros(self.get_anchor.shape[0], self.times, device=self.get_anchor.device)
        
        # 使用新的编码函数处理时间戳
        geo_encoded = self.encode_timestamp(geo_time_stamp, self.times)
        time_encoded = self.encode_timestamp(viewpoint_camera.time_stamp, self.times)

        anchor_geo_time_stamps_onehot[:] = geo_encoded
        anchor_time_stamps_onehot[:] = time_encoded
        
        anchor_geo_time_stamps_onehot = anchor_geo_time_stamps_onehot[visible_mask]
        anchor_time_stamps_onehot = anchor_time_stamps_onehot[visible_mask]


        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]

        if not hasattr(self, 'base_feat_dim'):
            self.base_feat_dim = self.feat_dim
        
        if viewpoint_camera.time_stamp is not None:
            feat = torch.zeros(self.get_anchor.shape[0], self.feat_dim, device=self.get_anchor.device)[visible_mask]
            geo_feat = torch.zeros(self.get_anchor.shape[0], self.feat_dim, device=self.get_anchor.device)[visible_mask]
            assert self.feat_dim == self.base_feat_dim + self.varing_feat_dim
            
            feat[:,:self.base_feat_dim] += self.get_anchor_feat[:,:self.base_feat_dim][visible_mask]
            geo_feat[:,:self.base_feat_dim] += self.get_anchor_feat[:,:self.base_feat_dim][visible_mask]
            
            for i in range(self.times):
                time_weight = anchor_time_stamps_onehot[:, i].unsqueeze(-1)  # [M, 1]
                geo_time_weight = anchor_geo_time_stamps_onehot[:, i].unsqueeze(-1)  # [M, 1]

                feat[:,self.base_feat_dim:] += self.get_anchor_feat[:, self.base_feat_dim + self.varing_feat_dim * (i): self.base_feat_dim + self.varing_feat_dim * (i+1)][visible_mask]  * time_weight
                geo_feat[:,self.base_feat_dim:] += self.get_anchor_feat[:, self.base_feat_dim + self.varing_feat_dim * (i): self.base_feat_dim + self.varing_feat_dim * (i+1)][visible_mask] * geo_time_weight
        else:
            feat = self.get_anchor_feat[visible_mask]

        grid_offsets = self.get_offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]

        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if self.use_feat_bank:
            bank_weight = self.get_featurebank_mlp(ob_view).unsqueeze(dim=1) # [n, 1, 3]

            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                feat[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1) # [n, c]

        cat_local_view = torch.cat([feat, ob_view], dim=1) # [N, c+3]
        geo_cat_local_view = torch.cat([geo_feat, ob_view], dim=1) # [N, c+3]

        if self.global_feat_dim > 0:
            if ape_code < 0:
                global_feat = torch.zeros(self.get_anchor.shape[0], self.global_feat_dim, device=self.get_anchor.device)[visible_mask]
                geo_global_feat = torch.zeros(self.get_anchor.shape[0], self.global_feat_dim, device=self.get_anchor.device)[visible_mask]
                for i in range(self.times):
                    global_feat += self.get_global_feat(torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * i) * anchor_time_stamps_onehot[:, i].unsqueeze(-1)
                    geo_global_feat += self.get_global_feat(torch.ones_like(geo_cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * i) * (anchor_geo_time_stamps_onehot[:, i].unsqueeze(-1))
            else:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
                global_feat = self.get_global_feat(camera_indicies)
                geo_camera_indicies = torch.ones_like(geo_cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
                geo_global_feat = self.get_global_feat(geo_camera_indicies)
        

        # get offset's opacity
        if hasattr(self, 'use_global_feat_in_opacity') and self.use_global_feat_in_opacity and self.global_feat_dim > 0:
            neural_opacity = self.get_opacity_mlp(torch.cat([geo_cat_local_view, geo_global_feat], dim=1))
        else:
            neural_opacity = self.get_opacity_mlp(geo_cat_local_view)


        # opacity mask generation
        ret_mask = (neural_opacity>0.0)
        ret_mask = ret_mask.reshape(-1)

        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.reshape(-1)

        # select opacity 
        opacity = neural_opacity[mask]

        # get offset's color
        if self.global_feat_dim > 0:
            color = self.get_color_mlp(torch.cat([cat_local_view, global_feat], dim=1))
        else:
            color = self.get_color_mlp(cat_local_view)
        
        color = color.reshape([anchor.shape[0]*self.n_offsets, 3])# [mask]

        # get offset's cov
        if hasattr(self, 'use_global_feat_in_cov') and self.use_global_feat_in_cov and self.global_feat_dim > 0:
            scale_rot = self.get_cov_mlp(torch.cat([geo_cat_local_view, geo_global_feat], dim=1))
        else:
            scale_rot = self.get_cov_mlp(geo_cat_local_view)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self.n_offsets, 7]) # [mask]
        # offsets
        offsets = grid_offsets.reshape([-1, 3]) # [mask]
        
        # combine for parallel masking
        
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]

        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
        
        # post-process cov
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
        rot = self.rotation_activation(scale_rot[:,3:7])
        
        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets 

        return xyz, color, opacity, scaling, rot, None, ret_mask

