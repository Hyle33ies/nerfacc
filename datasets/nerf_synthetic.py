"""
Dataset loader for NeRF Synthetic dataset.
"""

import json
import os
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SubjectLoader(Dataset):
    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        num_rays: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.split = split
        self.num_rays = num_rays
        self.device = device
        
        # Load transforms.json
        with open(os.path.join(root_fp, subject_id, "transforms_{}.json".format(split)), "r") as f:
            self.meta = json.load(f)
            
        self.images = []
        self.poses = []
        
        # Load images and poses
        for frame in self.meta["frames"]:
            fname = os.path.join(root_fp, subject_id, frame["file_path"] + ".png")
            image = cv2.imread(fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            self.images.append(image)
            self.poses.append(np.array(frame["transform_matrix"]))
            
        # Convert to tensors
        self.images = torch.from_numpy(np.stack(self.images)).float().to(device)
        self.poses = torch.from_numpy(np.stack(self.poses)).float().to(device)
        
        # Camera parameters
        self.H, self.W = self.images.shape[1:3]
        camera_angle_x = float(self.meta["camera_angle_x"])
        self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)
        
        # Ray directions
        i, j = torch.meshgrid(
            torch.arange(self.W, device=device),
            torch.arange(self.H, device=device),
            indexing="xy"
        )
        self.directions = torch.stack([
            (i - self.W * 0.5) / self.focal,
            -(j - self.H * 0.5) / self.focal,
            -torch.ones_like(i)
        ], dim=-1)

    def __len__(self):
        return len(self.images)

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        rays_o = pose[:3, -1].expand(self.H * self.W, 3)
        rays_d = (pose[:3, :3] @ self.directions.reshape(-1, 3).T).T

        if self.num_rays is not None:
            selected_inds = torch.randperm(self.H * self.W)[:self.num_rays]
            rays_o = rays_o[selected_inds]
            rays_d = rays_d[selected_inds]
            pixels = image.reshape(-1, 3)[selected_inds]
        else:
            pixels = image.reshape(-1, 3)

        return {
            "rays": torch.cat([rays_o, rays_d], dim=-1),
            "pixels": pixels,
            "color_bkgd": torch.ones(3, device=self.device),
        } 
