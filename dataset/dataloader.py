import os
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import h5py


class GTSamples(Dataset):
    """Dataset for training and testing with sketch images and occupancy data"""
    def __init__(self, data_source, partition="train", view_pick=False, balance=True, num_testing_points=2048, img_size=256, seed=42):
        super().__init__()
        self.data_source = data_source
        self.partition = partition
        self.view_pick = view_pick
        self.img_size = img_size
        self.seed = seed
        self.balance = balance
        self.num_testing_points = num_testing_points
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.edgemap_root = os.path.join(self.data_source, "renderingimg/edgemap")
        self.depthmap_root = os.path.join(self.data_source, "renderingimg/depthmap")
        self.normalmap_root = os.path.join(self.data_source, "renderingimg/normalmap")
        
        # Load HDF5 data and name files
        self._load_hdf5_data()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        self.normal_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        assert self.partition in ["train", "test"], "Partition must be 'train' or 'test'"

        # Get available image model IDs
        self._get_image_model_ids()
        
        # Build data paths with HDF5 index mapping
        self._build_data_paths()

    def _load_hdf5_data(self):
        """Load HDF5 data and create mapping between simplified and full IDs"""
        if self.partition == "test":
            hdf5_file = os.path.join(self.data_source, 'voxel2mesh.hdf5')
            name_file = os.path.join(self.data_source, 'test_names.npz')
            npz_data = np.load(name_file)
            self.hdf5_names = npz_data['test_names']
            self.hdf5_file = h5py.File(hdf5_file, 'r')
            self.hdf5_points = torch.from_numpy(self.hdf5_file['points'][:]).float()
            self.hdf5_points[:, :, :3] = (self.hdf5_points[:, :, :3] + 0.5) / 64 - 0.5
        else:  # train
            hdf5_file = os.path.join(self.data_source, 'ae_train.hdf5')
            name_file = os.path.join(self.data_source, 'train_names.npz')
            npz_data = np.load(name_file)
            self.hdf5_names = npz_data['train_names']
            self.hdf5_file = h5py.File(hdf5_file, 'r')
            self.hdf5_points = torch.from_numpy(self.hdf5_file['points_64'][:]).float()
        
        print(f'Loaded HDF5 points shape: {self.hdf5_points.shape}')
        print(f'Loaded {len(self.hdf5_names)} model names from {name_file}')
        
        # Create mapping from simplified ID to HDF5 index
        self.id_to_hdf5_idx = {}
        for idx, full_name in enumerate(self.hdf5_names):
            # Extract simplified ID (first 8 characters)
            if isinstance(full_name, bytes):
                simplified_id = full_name.decode('utf-8')[:8]
            else:
                simplified_id = str(full_name)[:8]
            
            if simplified_id not in self.id_to_hdf5_idx:
                self.id_to_hdf5_idx[simplified_id] = []
            self.id_to_hdf5_idx[simplified_id].append(idx)
        
        print(f'Created mapping for {len(self.id_to_hdf5_idx)} unique simplified IDs')

    def _get_image_model_ids(self):
        """Get available model IDs from image directories"""
        if not os.path.exists(self.edgemap_root):
            raise FileNotFoundError(f"Edge map root {self.edgemap_root} does not exist")
        
        # Get all available model IDs from edgemap directory
        available_model_ids = []
        for model_dir in sorted(os.listdir(self.edgemap_root)):
            model_path = os.path.join(self.edgemap_root, model_dir)
            if os.path.isdir(model_path):
                # Check if there are any PNG files in this directory
                png_files = glob.glob(os.path.join(model_path, "*.png"))
                if png_files:
                    available_model_ids.append(model_dir)
        
        self.available_model_ids = available_model_ids
        print(f'Found {len(self.available_model_ids)} available model IDs in image data')

    def _build_data_paths(self):
        """Build data paths by matching image model IDs with HDF5 data"""
        self.data_paths = []
        
        for model_id in self.available_model_ids:
            # Check if this model_id exists in HDF5 data
            if model_id not in self.id_to_hdf5_idx:
                # print(f"Warning: Model ID {model_id} not found in HDF5 data. Skipping.")
                continue
            
            edge_dir = os.path.join(self.edgemap_root, model_id)
            if not os.path.exists(edge_dir):
                print(f"Warning: Edge directory not found for {model_id}. Skipping.")
                continue
            
            edge_views = sorted(glob.glob(os.path.join(edge_dir, "*.png")))
            edge_views = [os.path.basename(v) for v in edge_views]
            if not edge_views:
                print(f"Warning: No edge views found for {model_id}. Skipping.")
                continue
            
            # Get HDF5 indices for this model_id (there might be multiple)
            hdf5_indices = self.id_to_hdf5_idx[model_id]
            
            if self.view_pick:
                # Pick one random view and one random HDF5 entry
                view_name = random.choice(edge_views)
                hdf5_idx = random.choice(hdf5_indices)
                view_id = view_name.split('_view_')[-1].split('.')[0]
                
                edge_path = os.path.join(edge_dir, view_name)
                depth_path = os.path.join(self.depthmap_root, model_id, f"{model_id}_view_{view_id}_depth.png")
                normal_path = os.path.join(self.normalmap_root, model_id, f"{model_id}_view_{view_id}_normal.png")
                
                if all(os.path.exists(p) for p in [edge_path, depth_path, normal_path]):
                    self.data_paths.append((edge_path, depth_path, normal_path, hdf5_idx, model_id, view_id))
            else:
                # Use all views with all HDF5 entries for this model
                for view_name in edge_views:
                    for hdf5_idx in hdf5_indices:
                        view_id = view_name.split('_view_')[-1].split('.')[0]
                        edge_path = os.path.join(edge_dir, view_name)
                        depth_path = os.path.join(self.depthmap_root, model_id, f"{model_id}_view_{view_id}_depth.png")
                        normal_path = os.path.join(self.normalmap_root, model_id, f"{model_id}_view_{view_id}_normal.png")
                        
                        if all(os.path.exists(p) for p in [edge_path, depth_path, normal_path]):
                            self.data_paths.append((edge_path, depth_path, normal_path, hdf5_idx, model_id, view_id))
        
        print(f'Built {len(self.data_paths)} data paths')
        
    def _save_image(self, image, filename):
        """Save a tensor or numpy array as an image file."""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.shape[0] in [1, 3]:  # [C, H, W]
            image = image.transpose(1, 2, 0)  # [H, W, C]
        if image.shape[-1] == 1:  # Grayscale
            image = image.squeeze(-1)
            plt.imsave(filename, image, cmap='gray')
        else:  # RGB
            plt.imsave(filename, image)
            
    def _visualize_points(self, points, occupancies, filename):
        """Visualize 3D points with occupancy=1 using matplotlib."""
        from mpl_toolkits.mplot3d import Axes3D
        occupied_points = points[occupancies == 1]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(occupied_points[:, 0], occupied_points[:, 1], occupied_points[:, 2], s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(filename)
        plt.close()

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        edge_path, depth_path, normal_path, hdf5_idx, model_id, view_id = self.data_paths[idx]

        # Load images
        sketch_img = Image.open(edge_path).convert('L')  # Grayscale
        depth_img = Image.open(depth_path).convert('L')  # Grayscale
        normal_img = Image.open(normal_path).convert('RGB')  # RGB

        # Apply transforms
        sketch_image = self.transform(sketch_img)
        depth_image = self.transform(depth_img)
        normal_image = self.normal_transform(normal_img)

        # Load occupancy data from HDF5 (no sampling, return all points)
        occ_points_data = self.hdf5_points[hdf5_idx]  # Shape: [N, 4] where last dim is [x, y, z, occupancy]
        
        points = occ_points_data[:, :3].numpy().astype(np.float32)  # [N, 3]
        occupancies = occ_points_data[:, 3].numpy().astype(np.float32)  # [N]
        
        # print(f"\nSample {idx}:")
        # print(f"  Model ID: {model_id}")
        # print(f"  View ID: {view_id}")
        # print(f"  HDF5 Index: {hdf5_idx}")
        # print(f"  Full Model Name: {str(self.hdf5_names[hdf5_idx])}")
        # print(f"  Sketch Image Shape: {sketch_image.shape}, Min: {sketch_image.min():.4f}, Max: {sketch_image.max():.4f}")
        # print(f"  Depth Image Shape: {depth_image.shape}, Min: {depth_image.min():.4f}, Max: {depth_image.max():.4f}")
        # print(f"  Normal Image Shape: {normal_image.shape}, Min: {normal_image.min():.4f}, Max: {normal_image.max():.4f}")
        # print(f"  Points Shape: {points.shape}, Min: {points.min(axis=0)}, Max: {points.max(axis=0)}")
        # print(f"  Occupancies Shape: {occupancies.shape}, Unique: {np.unique(occupancies)}, "
              # f"Num Occupied (1): {np.sum(occupancies == 1)}, Num Unoccupied (0): {np.sum(occupancies == 0)}")
              
        # os.makedirs("debug_images", exist_ok=True)
        # self._save_image(sketch_image, f"debug_images/sample_{idx}_sketch.png")
        # self._save_image(depth_image, f"debug_images/sample_{idx}_depth.png")
        # self._save_image(normal_image, f"debug_images/sample_{idx}_normal.png")
        # self._visualize_points(points, occupancies, f"debug_images/sample_{idx}_points.png")
        
        if self.balance:
            inner_indices = np.where(occupancies == 1)[0]
            outer_indices = np.where(occupancies == 0)[0]
            
            # Sample equal number of occupied and unoccupied points
            inner_sample_size = int(self.num_testing_points * 0.5)
            out_sample_size = self.num_testing_points - inner_sample_size
            
            # Handle empty indices
            if len(inner_indices) == 0 or len(outer_indices) == 0:
                # print(f"Warning: Empty indices for model {model_id}, hdf5_idx {hdf5_idx}. "
                      # f"Inner points: {len(inner_indices)}, Outer points: {len(outer_indices)}")
                # Fallback to random sampling
                indices = np.random.choice(len(points), self.num_testing_points, replace=len(points) < self.num_testing_points)
            else:
                inner_sample = np.random.choice(inner_indices, inner_sample_size, replace=len(inner_indices) < self.num_testing_points)
                outer_sample = np.random.choice(outer_indices, out_sample_size, replace=len(outer_indices) < self.num_testing_points)
            
                # Combine the two sets
                indices = np.concatenate([inner_sample, outer_sample])
            
                # Shuffle the indices
                np.random.shuffle(indices)
        else:
            # Random sampling without balancing
            indices = np.random.choice(len(points), self.num_testing_points, replace=len(points) < self.num_testing_points)

        points = points[indices].astype(np.float32)
        occupancies = occupancies[indices].astype(np.float32)

        return {
            'sketch_image': sketch_image,  # [C, H, W], C=1
            'depth_image': depth_image,    # [C, H, W], C=1
            'normal_image': normal_image,  # [C, H, W], C=3
            'points': points,              # [N, 3]
            'occupancies': occupancies,    # [N]
            'model_id': model_id,
            'view_id': view_id,
            'hdf5_idx': hdf5_idx,
            'full_model_name': str(self.hdf5_names[hdf5_idx])
        }

    def __del__(self):
        """Close HDF5 file when dataset is destroyed"""
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()


if __name__ == "__main__":
    train_dataset = GTSamples(data_source='/home/dd005/workspace/occnet/data/', partition='test', view_pick=True)
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        # Images are saved and printed in __getitem__
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    for batch in train_loader:
        print(f"\nBatch Info:")
        print(f"  Batch Sketch Image Shape: {batch['sketch_image'].shape}")
        print(f"  Batch Depth Image Shape: {batch['depth_image'].shape}")
        print(f"  Batch Normal Image Shape: {batch['normal_image'].shape}")
        print(f"depth_map: min={batch['depth_image'].min().item()}, max={batch['depth_image'].max().item()}")
        print(f"normal_map: min={batch['normal_image'].min().item()}, max={batch['normal_image'].max().item()}")
        print(f"  Batch Points Shape: {batch['points'].shape}")
        print(f"  Batch Occupancies Shape: {batch['occupancies'].shape}")
        print(f"  Batch Occupancies Unique: {np.unique(batch['occupancies'].numpy())}")
        print(f"  Batch Model IDs: {batch['model_id']}")
        print(f"  Batch View IDs: {batch['view_id']}")
        break
