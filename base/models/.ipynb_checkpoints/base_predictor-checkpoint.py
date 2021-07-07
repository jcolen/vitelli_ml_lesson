import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from base_datasets.base_dataset import RandomCrop, ToTensor

'''
Base predictor
'''
class BasePredictor(pl.LightningModule):
	def forward(self, x):
		pass
	
	def getxy(self, batch):
		pass

	def get_dataset(self, *args, **kwargs):
		pass

	def compute_loss(self, batch):
		x, y0 = self.getxy(batch)
		y = self(x)
		loss = self.loss(y, y0)
		return loss
	
	def on_fit_start(self):
		print(self.hparams)
	
	def training_step(self, batch, batch_idx):
		loss = self.compute_loss(batch)
		return {'loss': loss}

	def validation_step(self, batch, batch_idx):
		loss = self.compute_loss(batch)
		return {'loss': loss, 'batch': batch}
	
	def training_epoch_end(self, outs):
		avg_loss = torch.stack([x['loss'] for x in outs]).mean()
		self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
		if self.trainer.progress_bar_callback is None or \
		   self.trainer.progress_bar_callback.refresh_rate == 0:
			print('Epoch %d: Train Loss:\t\t%g' % (self.current_epoch, avg_loss))
	
	def validation_epoch_end(self, outs):
		avg_loss = torch.stack([x['loss'] for x in outs]).mean()
		self.logger.experiment.add_scalar('Loss/Validation', avg_loss, self.current_epoch)

		fig = self.predict_plot(outs[0]['batch'])	
		self.logger.experiment.add_figure('Figure', fig, close=True, global_step=self.trainer.global_step)

		self.logger.log_hyperparams(self.hparams, {'val_loss': avg_loss})
		if self.trainer.progress_bar_callback is None or \
		   self.trainer.progress_bar_callback.refresh_rate == 0:
			print('Epoch %d: Validation Loss:\t%g' % (self.current_epoch, avg_loss))

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.scheduler_step)
		return {'optimizer': optimizer,
				'lr_scheduler': scheduler}	

	def get_transform(self, crop_size):
		return transforms.Compose([
			RandomCrop(crop_size),
			ToTensor()
		])

	def predict_plot(self, batch, cmap='BuPu'):
		with torch.no_grad():
			x, y0 = self.getxy(batch)
			y = self(x)
			xval, xvec, y0val, y0vec, yval, yvec = self.convert_xy_visual(x, y0, y)
		nplots = min(len(next(iter(batch.values()))), 4)
		fig, ax = plt.subplots(nplots, 4, dpi=200)
		for i in range(nplots):
			for j in range(4):
				ax[i, j].clear()
				ax[i, j].set_xticks([])
				ax[i, j].set_yticks([])

			skip = 10
			xidx = (i, slice(None, None, skip), slice(None, None, skip), 0)
			yidx = (i, slice(None, None, skip), slice(None, None, skip), 1)

			def showquiver(ax, val, vec, cmap='BuPu'):
				if val is None:
					ax.set_facecolor('black')
					yi, xi = np.mgrid[:vec.shape[1]:skip, :vec.shape[2]:skip]
					ax.quiver(xi, yi, vec[xidx], vec[yidx], color='white')
				elif vec is None:
					ax.imshow(val[i], cmap=cmap)
				else:
					ax.imshow(val[i], cmap=cmap)
					yi, xi = np.mgrid[:vec.shape[1]:skip, :vec.shape[2]:skip]
					ax.quiver(xi, yi, vec[xidx], vec[yidx])

			showquiver(ax[i, 0], xval, xvec, cmap='winter')
			showquiver(ax[i, 1], y0val, y0vec, cmap=cmap)
			showquiver(ax[i, 2], yval, yvec, cmap=cmap)
			if yvec is not None:
				showquiver(ax[i, 3], np.linalg.norm(yvec - y0vec, axis=-1), None, cmap='jet')
			else:
				showquiver(ax[i, 3], abs(yval - y0val), None, cmap='jet')

		ax[0, 0].set_title('Input')
		ax[0, 1].set_title('Target')
		ax[0, 2].set_title('Predicted')
		ax[0, 3].set_title('Diff')
		
		plt.tight_layout()
		return fig
