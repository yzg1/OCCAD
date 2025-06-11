import torch
from .base import BaseTrainer
from .loss import reconLoss
from .acc_recall import acc_recall
from model import Encoder,Decoder,Generator
from collections import OrderedDict


class TrainerAE(BaseTrainer):
    """Trainer for training SECAD-Net.
    """
    def build_net(self):
        self.encoder = Encoder().cuda()
        self.decoder = Decoder(num_primitives=self.specs["NumPrimitives"]).cuda()
        self.generator = Generator(num_primitives=self.specs["NumPrimitives"]).cuda()
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.decoder = torch.nn.DataParallel(self.decoder)
            self.generator = torch.nn.DataParallel(self.generator)

    def set_optimizer(self, lr, betas):
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": lr, "betas": (betas[0], betas[1])},
                {"params": self.decoder.parameters(), "lr": lr, "betas": (betas[0], betas[1])},
                {"params": self.generator.parameters(), "lr": lr, "betas": (betas[0], betas[1])}
            ],
            weight_decay=1e-5
        )
        self.clip_gradient_norm = self.specs.get("ClipGradientNorm", 1.0)
    
    def set_loss_function(self):
        self.loss_func = reconLoss(self.specs["LossWeightTrain"]).cuda()

    def set_accuracy_function(self):
        self.acc_func = acc_recall().cuda()

    def forward(self, data):
        sketch_image = data['sketch_image'].cuda()  # [B, 1, H, W]
        gt_depth = data['depth_image'].cuda()       # [B, 1, H, W]
        gt_normal = data['normal_image'].cuda()     # [B, 3, H, W]
        xyz = data['points'].cuda()                 # [B, M, 3]
        gt_3d_occ = data['occupancies'].cuda()       # [B, M]

        shape_code, pred_depth, pred_normal = self.encoder(sketch_image)
        shape_3d = self.decoder(shape_code)
        output_3d_occ, total_2d_occ, transformed_points = self.generator(xyz, shape_3d, shape_code)
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
        
        del shape_code, shape_3d, transformed_points, h, xyz, gt_3d_occ
        torch.cuda.empty_cache()

        return outputs, loss_dict, acc_dict
    
    def train_func(self, data):
        """one step of training"""
        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        self.optimizer.zero_grad()
        outputs, losses, acc_recall = self.forward(data)
        total_loss = sum(losses.values())
        total_loss.backward()
        
        gt_depth = data['depth_image'].cuda()
        gt_normal = data['normal_image'].cuda()
        pred_depth = outputs['pred_depth']
        pred_normal = outputs['pred_normal']
        
        
        torch.nn.utils.clip_grad_norm_(
            parameters=[
                *self.encoder.parameters(),
                *self.decoder.parameters(),
                *self.generator.parameters()
            ],
            max_norm=self.clip_gradient_norm,
            norm_type=2.0
        )
        
        self.optimizer.step()
        
        if self.clock.step % 50 == 0:
            torch.cuda.empty_cache()
        
        self.update_epoch_info(losses, acc_recall)
        if self.clock.step % 10 == 0:
            self.record_to_tb(losses, acc_recall)
        
        loss_info = OrderedDict({k: "{:.6f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_loss.items()})
        acc_info = OrderedDict({k: "{:.2f}".format(v.item()/(self.clock.minibatch+1))
                                for k, v in self.epoch_acc.items()})
        out_info = loss_info.copy()
        out_info.update(acc_info)
        return outputs, out_info