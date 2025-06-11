import os
import torch
import argparse
from tqdm import tqdm
import torch.utils.data as data_utils

from utils import init_seeds
from utils.workspace import load_experiment_specifications
from trainer import FineTunerAE
from dataset import dataloader
import numpy as np


def main(args):
    # Set random seed
    init_seeds()
    
    # Load experiment specifications
    experiment_directory = os.path.join('./exp_log', args.experiment_directory)
    specs = load_experiment_specifications(experiment_directory)
    
    # Create dataset and data loader
    occ_dataset = dataloader.GTSamples(specs["DataSource"], partition="test", view_pick=True)
    
    if args.data_subset is None:
        print('Running the complete data set sequentially.')
        shape_indexes = list(range(int(args.start_index), int(args.end_index)))
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
                    print(f"{model_id} not found in model_ids.")
            else:
                print(f"{model_id} not found in available_model_ids.")
    
    print('Indices of shapes that need fine-tuning: ', shape_indexes)
    
    epoches_ft = int(args.epoches)
    
    specs["experiment_directory"] = experiment_directory
    
    for index in shape_indexes:
        print('Fine-tuning shape index', index)
        print(f"data_paths[{index}]: {occ_dataset.data_paths[index]}")
        shapename = occ_dataset.data_paths[index][4]
        data = occ_dataset[index]
        sketch_image = data['sketch_image'].unsqueeze(0).cuda()
        gt_depth = data['depth_image'].unsqueeze(0).cuda()      # [1, 1, H, W]
        gt_normal = data['normal_image'].unsqueeze(0).cuda()    # [1, 3, H, W]
        points = torch.from_numpy(data['points']).unsqueeze(0).cuda()
        occupancies = torch.from_numpy(data['occupancies']).unsqueeze(0).cuda()
        
        ft_agent = FineTunerAE(specs)
        start_epoch = ft_agent.load_shape_code(sketch_image, args.checkpoint)
        
        # start finetuning
        clock = ft_agent.clock
        pbar = tqdm(range(start_epoch, start_epoch + epoches_ft))
        
        for e in pbar:
            for i in range(40):
                occ_data = {
                    'sketch_image': sketch_image,
                    'gt_depth': gt_depth,
                    'gt_normal': gt_normal,
                    'points': points,
                    'occupancies': occupancies
                }
                outputs, out_info = ft_agent.train_func(occ_data)
                pbar.set_description("EPOCH[{}][{}]".format(e, epoches_ft))
                clock.tick()
            pbar.set_postfix(out_info)
            ft_agent.save_model_if_best_per_shape(shapename)
            clock.tock()
 
 
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
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
        default="best"
        )	
    arg_parser.add_argument(
        "--subset",
        dest="data_subset",
        default=None
        )
    arg_parser.add_argument(
        "--start",
        dest="start_index",
        default=0
        )
    arg_parser.add_argument(
        "--end",
        dest="end_index",
        default=1
        )
    arg_parser.add_argument(
        "--epoches",
        dest="epoches",
        default=300,
        )
    arg_parser.add_argument(
        "--gpu",
        "-g",
        dest="gpu",
        default=0,
        )
        
    args = arg_parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
    
    main(args)
