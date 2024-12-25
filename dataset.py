import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from point_transforms import *

class ShapeNet(data.Dataset):
    def __init__(self,
                 img_root='./data/renderingimg',
                 pc_occ_root='./data/pc_occ',
                 filelist_root='./data/train_val_test_list',
                 mode='train', view_pick=False,
                 occ_packbits=True, n_pc_occ_subsample=4096,
                 surface_ratio=0.5,
                 sigma_ratio=0.03,                 
                 occ_data_with_transforms=False):

        self.img_root = img_root
        self.pc_occ_root = pc_occ_root
        self.filelist_root = filelist_root
        self.mode = mode
        self.occ_packbits = occ_packbits
        self.occ_data_with_transforms = occ_data_with_transforms
        self.sampler = AdaptiveSurfaceSampler(num_samples=n_pc_occ_subsample, surface_ratio=surface_ratio, sigma_ratio=sigma_ratio)
        
        if mode == 'train':
            # self.pc_occ_transform = SubsamplePoints(n_pc_occ_subsample)
            self.pc_occ_transform = AdaptiveSurfaceSampler(num_samples=n_pc_occ_subsample, surface_ratio=surface_ratio, sigma_ratio=sigma_ratio)
        else:
            self.pc_occ_transform = None

        list_file = os.path.join(self.filelist_root, mode + '.txt')
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file {list_file} not found.")
        
        with open(list_file, 'r') as f:
            fnames = [line.strip() for line in f.readlines()]

        self.data_paths = []
        for file_name in fnames:
            points_path = os.path.join(self.pc_occ_root, file_name, 'points.npz')
            pointcloud_path = os.path.join(self.pc_occ_root, file_name, 'points.npz')
            img_view_path = os.path.join(self.img_root, 'edgemap', file_name)
            
            if not os.path.exists(points_path):
                print(f"Warning: Points file {points_path} not found. Skipping {file_name}.")
                continue
            
            if not os.path.exists(pointcloud_path):
                print(f"Warning: Pointcloud file {pointcloud_path} not found. Skipping {file_name}.")
                continue

            with open(os.path.join(img_view_path, 'renderings.txt'), 'r') as f:
                view_list = list(set(f.readlines()))
                if not view_pick:
                    for view_name in view_list:
                        view_name = view_name.strip()
                        img_path = os.path.join(img_view_path, view_name)
                        self.data_paths.append((img_path, points_path, file_name, view_name.split('.')[0]))
                else:
                    view_name = view_list[0].strip()
                    img_path = os.path.join(img_view_path, view_name)
                    self.data_paths.append((img_path, points_path, file_name, view_name.split('.')[0]))
        print(f"Total data paths (raw): {len(self.data_paths)}")
        self.data_paths = list(set(self.data_paths))
        print(f"Total data paths (unique): {len(self.data_paths)}")

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # Data Augmentation
            # transforms.RandomHorizontalFlip(),  # Random horizontal flip
            # transforms.RandomRotation(15),  # Random rotation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
            img_path, pc_occ_path, name, view_id = self.data_paths[index]
            img = Image.open(img_path).convert('L')
            # img = Image.open(img_path).convert('RGB')
            img_data = self.img_transform(img)

            points_path = os.path.join(os.path.dirname(pc_occ_path), 'points.npz')
            occupancies_path = os.path.join(os.path.dirname(pc_occ_path), 'points.npz')
        
            points_dict = np.load(points_path)
            occupancies_dict = np.load(occupancies_path)
        
            points = points_dict['points'].astype(np.float32)
            occupancies = occupancies_dict['occupancies']
            # inside_occupancies = occupancies_dict['inside_occupancies']
            # outside_occupancies = occupancies_dict['outside_occupancies']

            # occupancies = np.concatenate([inside_occupancies, outside_occupancies], axis=0)
            # Same packbits handling as before
            # if self.occ_packbits:
            #     if occupancies.dtype != np.uint8:
            #         occupancies = occupancies.astype(np.uint8)
            #     occupancies = np.unpackbits(occupancies)[:points.shape[0]]
            occupancies = occupancies.astype(np.float32)
        
            occ_data = {
                None: points,
                'occ': occupancies,
            }
        
            if self.occ_data_with_transforms:
                occ_data['loc'] = points_dict.get('loc', None).astype(np.float32)
                occ_data['scale'] = points_dict.get('scale', None).astype(np.float32)
        
            if self.pc_occ_transform is not None:
                occ_data = self.pc_occ_transform(occ_data)
        
            return img_data, occ_data, name, view_id

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    dataset = ShapeNet(mode='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    for i, data in enumerate(dataloader, 0):
        img, occ, name, view_id = data
        print(f"Name: {name}")
        print(f"View ID: {view_id}")
        print(f"Image Shape: {img.shape}")
        print(f"Points: {occ}")
        print(f"Points shape: {occ[None].shape}")
        print(f"Occupancies: {occ['occ']}")
        print(f"Occupancies shape: {occ['occ'].shape}")

        unique, counts = np.unique(occ['occ'], return_counts=True)
        occ_distribution = dict(zip(unique, counts))

        print("Occupancy Value Distribution:", occ_distribution)

        total = counts.sum()
        print("Percentage Distribution:", {k: v / total * 100 for k, v in occ_distribution.items()})

        break
