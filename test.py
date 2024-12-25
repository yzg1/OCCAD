import time
import argparse
import torch.backends.cudnn as cudnn
from dataset import *
from model import *
from common import *
from torchvision.utils import save_image
from generation import Generator3D
import torchvision.transforms as transforms
import os
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='./data', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--test', type=str, default='test', help='test results path')
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--refinement', type=int, default=30, help='number of mesh refinement steps')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
cudnn.benchmark = True

# Create testing dataloader
dataset_test = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                        pc_occ_root=os.path.join(opt.dataRoot, 'pc_occ'),
                        filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                        mode='test', view_pick=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset_test)
print('test set num', len_dataset)

model_path = opt.model
test_path = opt.test
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Load network
network = OccupancyNetwork()
network.cuda()

def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if any(key.startswith('module.') for key in checkpoint.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

model_path = os.path.join(opt.model, 'occnet_best.pt')
load_model_weights(network, model_path)
network.eval()

vox_res = 64

# Create generator with mesh generation parameters
# without refining results
# generator = Generator3D(network, threshold=0.2, refinement_step=0, resolution0=vox_res, upsampling_steps=0)

# with refining results
generator = Generator3D(network, threshold=0.2, refinement_step=opt.refinement, resolution0=vox_res, upsampling_steps=2, simplify_nfaces=5000)

pts = make_3d_grid((-0.5,)*3, (0.5,)*3, (vox_res,)*3).contiguous().view(1, -1, 3)
pts = pts.cuda()

'''
Start testing
'''
total_time = 0
test_num = 0
# Disable gradient calculation
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, _, name, view_id = data
        img = img.cuda()

        print(i, 'processing', name[0])
        start_time = time.time()
        
        pts_occ_val = network.predict(img, pts)
        print('pts_occ_val stats - min:', pts_occ_val.min(), 'max:', pts_occ_val.max(), 'mean:', pts_occ_val.mean())
        
        # Generate mesh
        mesh, stats = generator.generate_mesh(img)
        
        cost_time = time.time() - start_time
        print('time cost:', cost_time)
        
        # Compute mesh statistics
        print(f'Mesh info - vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}')
        print('Generation stats:', stats)

        if i > 0:
            total_time += cost_time
            test_num += 1

        # Write output
        # Save input image
        sample_name = os.path.dirname(name[0])
        img_name = os.path.basename(name[0])
        output_folder = os.path.join(test_path, sample_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Save image
        output_img_file = os.path.join(output_folder, f"{view_id[0]}.png")
        save_image(img.squeeze(0).cpu(), output_img_file)
        
        # Save mesh as OBJ
        output_mesh_file = os.path.join(output_folder, f"{view_id[0]}_mesh.obj")
        mesh.export(output_mesh_file)

        if i > 19:
            break

print('average time cost:', total_time / test_num)
print('Done!')