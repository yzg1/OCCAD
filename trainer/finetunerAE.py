import os
import torch
from collections import OrderedDict
from .base import BaseTrainer
from .loss import reconLoss
from .acc_recall import acc_recall
from model import Encoder,Decoder,Generator
from utils.workspace import get_model_params_dir,get_model_params_dir_shapename


class FineTunerAE(BaseTrainer):
    """Trainer for fine-tuning or testing SECAD-Net.
    """
    def build_net(self):
        self.encoder = Encoder().cuda()
        self.decoder = Decoder(num_primitives=self.specs["NumPrimitives"]).cuda()
        self.generator = Generator(num_primitives=self.specs["NumPrimitives"]).cuda()

    def set_optimizer(self, lr, betas):
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.decoder.parameters(), "lr": lr, "betas": (betas[0], betas[1])},
                {"params": self.generator.parameters(), "lr": lr, "betas": (betas[0], betas[1])}
            ],
            weight_decay=1e-5
        )
        
    def set_loss_function(self):
        self.loss_func = reconLoss(self.specs["LossWeightFineTune"]).cuda()

    def set_accuracy_function(self):
        self.acc_func = acc_recall().cuda()
        
    def save_model_if_best_per_shape(self, shapename):
        epoch_loss_value = sum(self.epoch_loss.values()).item()/(self.clock.minibatch+1)
        if epoch_loss_value < self.best_loss:
            model_params_dir = get_model_params_dir(self.experiment_directory)
            model_params_dir = get_model_params_dir_shapename(model_params_dir, shapename)
            torch.save(
                {"epoch": self.clock.epoch,
                "shape_code_state_dict": self.shape_code,
                "decoder_state_dict": self.decoder.state_dict(),
                "generator_state_dict": self.generator.state_dict(),
                "opt_state_dict": self.optimizer.state_dict()}, 
                os.path.join(model_params_dir, 'best.pth')
            )
            self.best_loss = epoch_loss_value
    
    def forward(self, data):
        sketch_image = data['sketch_image']  # [B, 1, H, W]
        gt_depth = data['gt_depth']         # [B, 1, H, W]
        gt_normal = data['gt_normal']       # [B, 3, H, W]
        xyz = data['points']                # [B, M, 3]
        gt_3d_occ = data['occupancies']     # [B, M]
        
        shape_code, pred_depth, pred_normal = self.encoder(sketch_image)
        shape_3d = self.decoder(self.shape_code.cuda())
        output_3d_occ, total_2d_occ, transformed_points = self.generator(xyz, shape_3d, self.shape_code.cuda())
        h = shape_3d[:, 7, :].unsqueeze(1)

        outputs = {
            "output_3d_occ": output_3d_occ,
            "total_2d_occ": total_2d_occ,
            "transformed_points": transformed_points,
            "h": h,
            "pred_depth": pred_depth,
            "pred_normal": pred_normal
        }
        loss_dict = self.loss_func(outputs, gt_3d_occ, pred_depth, gt_depth, pred_normal, gt_normal)
        acc_dict = self.acc_func(outputs, gt_3d_occ)
        
        return outputs, loss_dict, acc_dict
    
    def load_shape_code(self, sketch_image, checkpoint):
        continue_from = checkpoint
        print('Continuing from "{}"'.format(continue_from))
        model_epoch = super().load_model_parameters(continue_from)
        with torch.no_grad():
            shape_code, _, _ = self.encoder(sketch_image)
        shape_code = shape_code.detach().cpu().numpy()

        shape_code = torch.from_numpy(shape_code)
        print('shape_code loaded, ', shape_code.shape)
        start_epoch = model_epoch +1 
            
        self.shape_code = shape_code
        self.shape_code.requires_grad = True
        
        self.optimizer_code = torch.optim.Adam(
            [
                {
                    "params": self.shape_code,
                    "lr": self.specs["LearningRate"],
                    "betas": (0.5, 0.999),
                },
            ]
        )
        print("Starting from epoch {}".format(start_epoch))
        return start_epoch
    
    def train_func(self, data):
        """one step of training"""
        self.decoder.train()
        self.generator.train()
        self.optimizer.zero_grad()
        self.optimizer_code.zero_grad()
        outputs, losses, acc_recall = self.forward(data)
        self.update_network(losses)
        self.update_epoch_info(losses, acc_recall)
        
        loss_info = OrderedDict({k: "{:.6f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_loss.items()})
        acc_info = OrderedDict({k: "{:.2f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_acc.items()})
        
        out_info = loss_info.copy()
        out_info.update(acc_info)
        return outputs, out_info
    
    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()
        self.optimizer_code.step()
        
    def evaluate(self, shapename, checkpoint):
        saved_model_epoch, shape_code = super().load_model_parameters_per_shape(
                shapename, checkpoint,
            )

        print('Loaded epoch: %d'%(saved_model_epoch))
        self.decoder.eval()
        self.generator.eval()

        shape_3d = self.decoder(shape_code.cuda())
        return shape_code, shape_3d
