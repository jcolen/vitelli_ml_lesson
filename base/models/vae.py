import torch
import torch.nn as nn
from base_models.conv_layers import RepeatFlatCnnCell, DownsampleCell, UpsampleCell
from base_models.base_predictor import BasePredictor
from base_models.loss import BetaVAELoss, JointLoss

'''
Generic convolutional VAE 
'''
class VAE(BasePredictor):
	def __init__(self, channels, repeat=2, beta=0.0001, **kwargs):
		super(VAE, self).__init__()
		self.channels = channels

		self.encoder = nn.ModuleList()

		for i in range(len(channels) - 1):
			self.encoder.append(RepeatFlatCnnCell(channels[i], channels[i+1], repeat=repeat))
			self.encoder.append(DownsampleCell(channels[i+1], channels[i+1]))
		
		self.en_to_param = RepeatFlatCnnCell(channels[-1], channels[-1], repeat=repeat)
		self.en_to_logvar = RepeatFlatCnnCell(channels[-1], channels[-1], repeat=repeat)

		self.decoder = nn.ModuleList()
		for i in range(1, len(channels)):
			self.decoder.append(RepeatFlatCnnCell(channels[-i], channels[-(i+1)], repeat=repeat))
			self.decoder.append(UpsampleCell(channels[-(i+1)], channels[-(i+1)]))
		
		self.loss = BetaVAELoss(beta=kwargs.get('beta', 0.0001))
		self.save_hyperparameters('channels', 'repeat', 'batch_size', 'crop_size', 
								  'learning_rate', 'scheduler_step', 'directory', 'beta')
	
	def forward(self, x):
		for i in range(len(self.encoder)):
			x = self.encoder[i](x)

		params = self.en_to_param(x)
		logvar = self.en_to_logvar(x)

		z = params
		if self.training:
			stdv = (0.5 * logvar).exp()
			z = z + stdv * torch.randn_like(z)

		for cell in self.decoder:
			z = cell(z)

		return z, params, logvar

'''
Convolutional VAE with two encoders sharing one decoder
'''
class MultiVAE(BasePredictor):
	def __init__(self, channels, repeat=2, **kwargs):
		super(MultiVAE, self).__init__()
		self.channels = channels

		self.encoder1 = nn.ModuleList()
		self.encoder2 = nn.ModuleList()

		for i in range(len(channels) - 1):
			self.encoder1.append(RepeatFlatCnnCell(channels[i], channels[i+1], repeat=repeat))
			self.encoder1.append(DownsampleCell(channels[i+1], channels[i+1]))
			self.encoder2.append(RepeatFlatCnnCell(channels[i], channels[i+1], repeat=repeat))
			self.encoder2.append(DownsampleCell(channels[i+1], channels[i+1]))
		
		self.e1_to_param = RepeatFlatCnnCell(channels[-1], channels[-1], repeat=repeat)
		self.e1_to_logvar = RepeatFlatCnnCell(channels[-1], channels[-1], repeat=repeat)
		self.e2_to_param = RepeatFlatCnnCell(channels[-1], channels[-1], repeat=repeat)
		self.e2_to_logvar = RepeatFlatCnnCell(channels[-1], channels[-1], repeat=repeat)

		self.decoder = nn.ModuleList()
		for i in range(1, len(channels)):
			self.decoder.append(RepeatFlatCnnCell(channels[-i], channels[-(i+1)], repeat=repeat))
			self.decoder.append(UpsampleCell(channels[-(i+1)], channels[-(i+1)]))
		
		self.loss = JointLoss(BetaVAELoss(beta=kwargs.get('beta', 0.0001)))
		self.save_hyperparameters('channels', 'repeat', 'batch_size', 'crop_size', 
								  'learning_rate', 'scheduler_step', 'directory', 'beta')
	
	def forward(self, x1, x2):
		for i in range(len(self.encoder1)):
			x1 = self.encoder1[i](x1)
			x2 = self.encoder2[i](x2)

		params1 = self.e1_to_param(x1)
		logvar1 = self.e1_to_logvar(x1)
		params2 = self.e2_to_param(x2)
		logvar2 = self.e2_to_logvar(x2)

		z1 = params1
		z2 = params2
		if self.training:
			stdv1 = (0.5 * logvar1).exp()
			stdv2 = (0.5 * logvar2).exp()
			z1 = z1 + stdv1 * torch.randn_like(z1)
			z2 = z2 + stdv2 * torch.randn_like(z2)

		for cell in self.decoder:
			z1 = cell(z1)
			z2 = cell(z2)

		return (z1, params1, logvar1), (z2, params2, logvar2)
