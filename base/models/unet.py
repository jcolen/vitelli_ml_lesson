import torch
import torch.nn as nn
import torch.nn.functional as F

from base.models.conv_layers import FlatCnnCell, DownsampleCell, UpsampleCell, RepeatFlatCnnCell
from base.models.base_predictor import BasePredictor

'''
Generic convolutional unet predictor
'''
class Unet(BasePredictor):
	def __init__(self, channels, repeat=2, **kwargs):
		super(Unet, self).__init__()
		self.channels = channels

		self.cells = nn.ModuleList()
		for i in range(len(channels)-1):
			self.cells.append(DownsampleCell(channels[i], channels[i+1]))
			self.cells.append(RepeatFlatCnnCell(channels[i+1], channels[i+1], 
				repeat=repeat, residual=False))

		for i in range(1, len(channels)):
			self.cells.append(UpsampleCell(channels[-i], channels[-(i+1)]))
			self.cells.append(FlatCnnCell(2*channels[-(i+1)], channels[-(i+1)]))
			self.cells.append(RepeatFlatCnnCell(channels[-(i+1)], channels[-(i+1)], 
				repeat=repeat, residual=False))
		
		self.save_hyperparameters('channels', 'repeat', 'batch_size', 'crop_size', 'learning_rate', 'scheduler_step', 'directory')

	def forward(self, x):
		encoder_outputs = []
		decoder_idx = -1
		for i, cell in enumerate(self.cells):
			if isinstance(cell, DownsampleCell):
				encoder_outputs.append(x)
			x = cell(x)
			if isinstance(cell, UpsampleCell):
				x2 = encoder_outputs[decoder_idx]
				diffY = x2.size()[-2] - x.size()[-2]
				diffX = x2.size()[-1] - x.size()[-1]
				x = F.pad(x, [diffX // 2, diffX - diffX // 2,
							  diffY // 2, diffY - diffY // 2])
				x = torch.cat([x, x2], dim=1)
				decoder_idx -= 1
		return x

