import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.sdfs import sdfExtrusion, transform_points
from utils.utils import add_latent


class UNetDepthNormalPredictor(nn.Module):
    def __init__(self, input_channels=1, output_resolution=256, features=32):
        super().__init__()
        self.output_resolution = output_resolution
        self.features = features
        
        # Encoder
        self.enc1 = self._double_conv(input_channels, features)
        self.enc2 = self._double_conv(features, features * 2)
        self.enc3 = self._double_conv(features * 2, features * 4)
        self.enc4 = self._double_conv(features * 4, features * 8)
        self.enc5 = self._double_conv(features * 8, features * 16)
        
        # Decoder for depth map
        self.up_d1 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec_d1 = self._double_conv(features * 16, features * 8)
        self.up_d2 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec_d2 = self._double_conv(features * 8, features * 4)
        self.up_d3 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec_d3 = self._double_conv(features * 4, features * 2)
        self.up_d4 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec_d4 = self._double_conv(features * 2, features)
        self.depth_output = nn.Conv2d(features, 1, kernel_size=1)
        
        # Decoder for normal map
        self.up_n1 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec_n1 = self._double_conv(features * 16, features * 8)
        self.up_n2 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec_n2 = self._double_conv(features * 8, features * 4)
        self.up_n3 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec_n3 = self._double_conv(features * 4, features * 2)
        self.up_n4 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec_n4 = self._double_conv(features * 2, features)
        self.normal_output = nn.Conv2d(features, 3, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._initialize_weights()
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        if x.shape[2:] != (self.output_resolution, self.output_resolution):
            x = F.interpolate(x, size=(self.output_resolution, self.output_resolution), mode='bilinear', align_corners=True)
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        
        # Depth map path
        d = self.up_d1(enc5)
        d = torch.cat([d, enc4], dim=1)
        d = self.dec_d1(d)
        d = self.up_d2(d)
        d = torch.cat([d, enc3], dim=1)
        d = self.dec_d2(d)
        d = self.up_d3(d)
        d = torch.cat([d, enc2], dim=1)
        d = self.dec_d3(d)
        d = self.up_d4(d)
        d = torch.cat([d, enc1], dim=1)
        d = self.dec_d4(d)
        depth_logits = self.depth_output(d)
        depth_map = torch.sigmoid(depth_logits)
        
        # Normal map path
        n = self.up_n1(enc5)
        n = torch.cat([n, enc4], dim=1)
        n = self.dec_n1(n)
        n = self.up_n2(n)
        n = torch.cat([n, enc3], dim=1)
        n = self.dec_n2(n)
        n = self.up_n3(n)
        n = torch.cat([n, enc2], dim=1)
        n = self.dec_n3(n)
        n = self.up_n4(n)
        n = torch.cat([n, enc1], dim=1)
        n = self.dec_n4(n)
        normal_logits = self.normal_output(n)
        normal_map = torch.tanh(normal_logits)
        
        return depth_map, normal_map

class Encoder(nn.Module):
    def __init__(self, ef_dim=32, input_resolution=256):
        super(Encoder, self).__init__()
        self.ef_dim = ef_dim
        self.input_resolution = input_resolution
        
        self.unet_predictor = UNetDepthNormalPredictor(input_channels=1, output_resolution=input_resolution)
        
        # Input [B, 5, 256, 256]
        self.conv_1 = nn.Conv2d(5, self.ef_dim, kernel_size=4, stride=2, padding=1, bias=True)                 # [B, 32, 128, 128]
        self.conv_2 = nn.Conv2d(self.ef_dim, self.ef_dim*2, kernel_size=4, stride=2, padding=1, bias=True)     # [B, 64, 64, 64]
        self.conv_3 = nn.Conv2d(self.ef_dim*2, self.ef_dim*4, kernel_size=4, stride=2, padding=1, bias=True)   # [B, 128, 32, 32]
        self.conv_4 = nn.Conv2d(self.ef_dim*4, self.ef_dim*8, kernel_size=4, stride=2, padding=1, bias=True)   # [B, 256, 16, 16]
        self.conv_5 = nn.Conv2d(self.ef_dim*8, self.ef_dim*8, kernel_size=4, stride=1, padding=0, bias=True)   # [B, 256, 13, 13]
        
        # [B, 256, 1, 1]
        self.fc = nn.Linear(256 * 13 * 13, self.ef_dim * 8, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for conv in [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, sketch_image):
        B = sketch_image.shape[0]
        pred_depth, pred_normal = self.unet_predictor(sketch_image)
        x = torch.cat([sketch_image, pred_depth, pred_normal], dim=1)
        
        for i in range(1, 6):
            x = getattr(self, f"conv_{i}")(x)
            x = F.leaky_relu(x, negative_slope=0.01)
        
        x = x.view(B, -1)  # [B, 256, 1, 1]
        x = self.fc(x)     # [B, 256]
        x = F.leaky_relu(x, negative_slope=0.01)
        
        return x, pred_depth, pred_normal


class Decoder(nn.Module):
    
	def __init__(self, ef_dim=32, num_primitives=4):
		super(Decoder, self).__init__()
		self.num_primitives = num_primitives
		self.feature_dim = ef_dim
  
		self.num_primitive_parameters_aggregated = 4+3+1
		self.primitive_linear = nn.Linear(self.feature_dim*8, 
                                    int(self.num_primitives*self.num_primitive_parameters_aggregated), 
                                    bias=True)
		nn.init.xavier_uniform_(self.primitive_linear.weight)
		nn.init.constant_(self.primitive_linear.bias, 0)

	def forward(self, feature):
		shapes = self.primitive_linear(feature)
		para_3d = shapes[...,:self.num_primitives*(4+3+1)].view(-1, (4+3+1), int(self.num_primitives)) # B,C,P
		return para_3d

class SketchHead(nn.Module):
	def __init__(self, d_in, dims):
		super().__init__()
		dims = [d_in] + dims + [1]
		self.num_layers = len(dims)
		for layer in range(0, self.num_layers - 1):
			out_dim = dims[layer + 1]
			lin = nn.Linear(dims[layer], out_dim)
   
			if layer == self.num_layers - 2:
				torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
				torch.nn.init.constant_(lin.bias, -1)
			else:
				torch.nn.init.constant_(lin.bias, 0.0)
				torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
			setattr(self, "lin" + str(layer), lin)
			self.activation = nn.Softplus(beta=100)
   
	def forward(self, input):
		x = input
		for layer in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(layer))
			x = lin(x)
			if layer < self.num_layers - 2:
				x = self.activation(x)
			else:
				x = x.clamp(-1,1)
		return x

class Generator(nn.Module):
	def __init__(self, num_primitives=4, sharpness=25.0, test=False):
		super(Generator, self).__init__()
		self.num_primitives = num_primitives
		self.sharpness = sharpness
		self.test=test
		D_IN = 2
		LATENT_SIZE = 256
		for i in range(num_primitives):
			setattr(self, 'sketch_head_'+str(i),
           			SketchHead(d_in=D_IN+LATENT_SIZE, dims = [ 512, 512, 512 ]),
           			)

	def forward(self, sample_point_coordinates, primitive_parameters, code):
		B, N = sample_point_coordinates.shape[:2]  # Batch size, number of testing points
		primitive_parameters = primitive_parameters.transpose(2, 1)
		B, K, param_dim = primitive_parameters.shape

		boxes = primitive_parameters[..., :8]
		transformed_points = transform_points(boxes[..., :4], boxes[..., 4:7], sample_point_coordinates)  # [B, N, K, 3]

		latent_points = []
		for i in range(self.num_primitives):
			latent_point = add_latent(transformed_points[..., i, :2], code).float()
			latent_points.append(latent_point)
		sdfs_2d = [getattr(self, f'sketch_head_{i}')(latent_points[i]).reshape(B, N, -1).float()\
      														for i in range(self.num_primitives)]
		sdfs_2d = torch.cat(sdfs_2d, dim=-1)

		total_2d_occ = torch.sigmoid(-1*sdfs_2d*self.sharpness)
  
		box_ext = sdfExtrusion(sdfs_2d, boxes[..., 7], transformed_points).squeeze(-1)
		primitive_sdf = box_ext
		primitive_occupancies = torch.sigmoid(-1*primitive_sdf*self.sharpness)

		with torch.no_grad():
			weights = torch.softmax(primitive_occupancies * 20, dim=-1)
		union_occupancies = torch.sum(weights * primitive_occupancies, dim=-1)
		return union_occupancies, total_2d_occ, transformed_points


