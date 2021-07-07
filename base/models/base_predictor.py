import torch
from pytorch_lightning import LightningModule

'''
Base predictor
'''
class BasePredictor(LightningModule):
	def forward(self, x):
		'''
		Compute an output from an input
		'''
		raise NotImplementedError()
	
	def get_transform(self):
		'''
		Define transformation pipeline for model
		'''
		raise NotImplementedError()
	
	def getxy(self, batch):
		'''
		Define how to get input/output pair from Dataset output
		'''
		raise NotImplementedError()
	
	def predict_plot(self, batch):
		'''
		(Optional) create a figure for logging/monitoring purposes
		'''
		raise NotImplementedError()

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

		try:
			fig = self.predict_plot(outs[0]['batch'])	
			self.logger.experiment.add_figure('Figure', fig, close=True, 
											  global_step=self.trainer.global_step)
		except:
			pass

		self.logger.log_hyperparams(self.hparams, {'val_loss': avg_loss})
		if self.trainer.progress_bar_callback is None or \
		   self.trainer.progress_bar_callback.refresh_rate == 0:
			print('Epoch %d: Validation Loss:\t%g' % (self.current_epoch, avg_loss))

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.scheduler_step)
		return {'optimizer': optimizer,
				'lr_scheduler': scheduler}	
