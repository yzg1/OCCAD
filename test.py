import os
import torch
import utils
import argparse
from utils.workspace import load_experiment_specifications
from torchvision.utils import save_image
from trainer import FineTunerAE
from dataset import dataloader
import numpy as np

def main(args):
    # Create experiment directory path
    experiment_directory = os.path.join('./exp_log', args.experiment_directory)
    
    # Load experiment specifications
    specs = load_experiment_specifications(experiment_directory)
    
    occ_dataset = dataloader.GTSamples(specs["DataSource"], partition="test", view_pick=True, balance=True)	
    
    reconstruction_dir = os.path.join(experiment_directory, "Reconstructions")
    MC_dir = os.path.join(reconstruction_dir, 'MC/')  # Dir for marching cube results
    CAD_dir = os.path.join(reconstruction_dir, 'CAD/')  # Dir for sketch-extrude results
    sk_dir = os.path.join(reconstruction_dir, 'sk/')  # Dir for 2d sketch images
    input_dir = os.path.join(reconstruction_dir, 'input/')
    depth_dir = os.path.join(reconstruction_dir, 'depth/')
    normal_dir = os.path.join(reconstruction_dir, 'normal/')
    
    
    for directory in [reconstruction_dir, CAD_dir, sk_dir, MC_dir, input_dir, depth_dir, normal_dir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        
    if args.data_subset is None:
        print('Running the complete data set sequentially.')
        shape_indexes = list(range(int(args.start), int(args.end)))
    else:
        print('Running on the specified data subset.')
        shape_indexes = []
        with open(args.data_subset, 'r') as file:
            eval_model_ids = [shape_name.strip() for shape_name in file]
        
        for model_id in eval_model_ids:
            if model_id in occ_dataset.available_model_ids:
                for dp_idx, data_path in enumerate(occ_dataset.data_paths):
                    if data_path[4] == model_id:
                        shape_indexes.append(dp_idx)
                        break
                else:
                    print(f"{model_id} not found in data_paths.")
            else:
                print(f"{model_id} not found in available_model_ids.")
    print('Shape indexes all: ', shape_indexes)
    
    specs["experiment_directory"] = experiment_directory
    ft_agent = FineTunerAE(specs)
    
    # Load the global model (includes the encoder)
    ft_agent.load_model_parameters("best")
    
    for index in shape_indexes:
        shapename = occ_dataset.data_paths[index][4]  # e.g., '00023457'
        data = occ_dataset[index]
        sketch_image = data['sketch_image'].unsqueeze(0).cuda()
        
        # Load shape-specific fine-tuned model and optimized shape code
        epoch, shape_code = ft_agent.load_model_parameters_per_shape(shapename, "best")
        shape_code = shape_code.cuda() # Ensure correct device
        
        with torch.no_grad():
            # Use fine-tuned decoder with optimized shape code
            shape_3d = ft_agent.decoder(shape_code)
            # Predict depth and normal using global encoder (for visualization)
            _, pred_depth, pred_normal = ft_agent.encoder(sketch_image)
            
        mesh_filename = os.path.join(MC_dir, shapename)
        CAD_mesh_filepath = os.path.join(CAD_dir, shapename)
        sk_filepath = os.path.join(sk_dir, shapename)
        
        # Create CAD mesh
        utils.create_CAD_mesh(ft_agent.generator, shape_code.cuda(), shape_3d.cuda(), CAD_mesh_filepath)
        
        # Create mesh using marching cubes
        utils.create_mesh_mc(ft_agent.generator, shape_3d.cuda(), shape_code.cuda(), mesh_filename, N=int(args.grid_sample), threshold=float(args.mc_threshold))
        
        # Draw 2D sketch image
        utils.draw_2d_im_sketch(shape_code.cuda(), ft_agent.generator, sk_filepath)
        
        sketch_image_cpu = sketch_image.cpu()
        save_image(sketch_image_cpu, os.path.join(input_dir, f"{shapename}_input.png"), normalize=False)
        
        pred_depth_cpu = pred_depth.cpu()
        save_image(pred_depth_cpu, os.path.join(depth_dir, f"{shapename}_depth.png"), normalize=True)
        
        pred_normal_cpu = pred_normal.cpu()
        normal_save = (pred_normal_cpu + 1) / 2
        save_image(normal_save, os.path.join(normal_dir, f"{shapename}_normal.png"))
        
        print(f"Predicted Depth: min={pred_depth.min().item():.4f}, max={pred_depth.max().item():.4f}")
        print(f"Predicted Normal: min={pred_normal.min().item():.4f}, max={pred_normal.max().item():.4f}")
        


if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(
		description="test trained model"
	)
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True
	)
	arg_parser.add_argument(
		"--checkpoint",
		"-c",
		dest="checkpoint",
		default="last_0"
	)
	arg_parser.add_argument(
		"--subset",
		dest="data_subset",
		default=None
		)
	arg_parser.add_argument(
		"--start",
		dest="start",
		default=0,
		help="start shape index",
	)
	arg_parser.add_argument(
		"--end",
		dest="end",
		default=1,
		help="end shape index",
	)
	arg_parser.add_argument(
		"--mc_threshold",
		dest="mc_threshold",
		default=0.5,
		help="marching cube threshold",
	)
	arg_parser.add_argument(
		"--gpu",
		"-g",
		dest="gpu",
		required=True,
		help="gpu id",
	)
	arg_parser.add_argument(
		"--grid_sample",
		dest="grid_sample",
		default=128,
		help="sample points resolution option",
	)
	args = arg_parser.parse_args()
 
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
 
	main(args)
	