import pytorch_lightning as pl

def train_model(model, dataset, args):
	batch_size = vars(args).get('batch_size', 32)
	num_workers = vars(args).get('num_workers', 2)
	train_loader = dataset.get_loader(dataset.train_indices, batch_size, num_workers)
	validation_loader = dataset.get_loader(dataset.validation_indices, batch_size, num_workers)

	logger = pl.loggers.TensorBoardLogger('tb_logs', name=model.name)
	trainer = pl.Trainer.from_argparse_args(args)
	trainer.logger = logger
	trainer.log_every_n_steps = min(len(train_loader), 50)
	trainer.fit(model, train_loader, validation_loader)
