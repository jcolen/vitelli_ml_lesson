import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from base.models.base_predictor import BasePredictor
from base.models.unet import Unet
from base.datasets.base_dataset import RandomFlip, RandomTranspose, RandomCrop, ToTensor
from dataset import ProcessInputs

class BaseModel(BasePredictor):
	def getxy(self, batch):
		return batch['fluorescence'], batch['label']
	
	def get_transform(self, crop_size, prob=0.5):
		return transforms.Compose([
			ProcessInputs(),
			RandomFlip(prob),
			RandomTranspose(prob),
			RandomCrop(crop_size),
			ToTensor()])
	
	def show_defects(self, ax, x, y):
		if len(x.shape) == 3:
			x = x[0]
		ax.clear()
		ax.set_xticks([])
		ax.set_yticks([])
		#Labels are 0 -> -1/2, 1 -> 0, 1 -> +1/2
		ax.imshow(x, cmap='Greys')
		rgba = np.zeros(x.shape + (4,))
		rgba[y == 0, :] = [1., 0, 0, 0.8]
		rgba[y == 2, :] = [0, 0.5, 1., 0.8]
		ax.imshow(rgba)

	def predict_plot(self, batch):
		with torch.no_grad():
			x, y0 = self.getxy(batch)
			y = torch.argmax(self(x), dim=1).cpu().numpy()
			x = x.cpu().numpy()
			y0 = y0.cpu().numpy()

		nplots = min(y0.shape[0], 4)
		fig, ax = plt.subplots(nplots, 2)
		ax[0, 0].set_title('Target')
		ax[0, 1].set_title('ML')
		for i in range(nplots):
			self.show_defects(ax[i, 0], x[i], y0[i])
			self.show_defects(ax[i, 1], x[i], y[i])
		plt.tight_layout()
		return fig

class UnetPredictor(BaseModel, Unet):
	'''
	Wrapper around UnetPredictor from base/models
	In this case, the input field has 1 channel and the output has 3
	So we need to define readin/readout layers to ensure the correct number of channels are there
	'''
	def __init__(self, channels, repeat=2, **kwargs):
		Unet.__init__(self, channels, repeat=repeat, **kwargs)
		self.loss = torch.nn.CrossEntropyLoss()

		self.read_in = RepeatFlatCnnCell(1, channels[0], repeat=repeat)
		self.read_out = RepeatFlatCnnCell(channels[0], 3, repeat=repeat)

		self.name = 'defect_unet'
	
	def forward(self, x):
		x = self.read_in(x)
		x = Unet.forward(self, x)
		x = self.read_out(x)
		return x
	
	def getxy(self, batch):
		return BaseModel.getxy(self, batch):
	
	def predict_plot(self, batch):
		return BaseModel.predict_plot(self, batch)

	

class fcn_resnet50(BaseModel):
	'''
	Wrapper around fcn_resnet50 from torchvision
	'''
	def __init__(self, **kwargs):
		super(fcn_resnet50, self).__init__()
		self.model = torchvision.models.segmentation.fcn_resnet50(num_classes=3, pretrained=False)
		self.name = 'fcn_resnet50'
		self.loss = torch.nn.CrossEntropyLoss()

	def forward(self, x):
		return self.model(x.repeat(1, 3, 1, 1))['out']
	
class fcn_resnet101(BaseModel):
	'''
	Wrapper around fcn_resnet101 from torchvision
	'''
	def __init__(self, **kwargs):
		BasePredictor.__init__(self)
		super(fcn_resnet101, self).__init__()
		self.name = 'fcn_resnet101'
		self.model = torchvision.models.segmentation.fcn_resnet101(num_classes=3, pretrained=False)
		self.loss = torch.nn.CrossEntropyLoss()
	
	def forward(self, x):
		return self.model(x.repeat(1, 3, 1, 1))['out']
