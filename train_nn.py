import matplotlib 
matplotlib.use('Agg')
import argparse
import pytorch_lightning as pl

from base.train import train_model
from dataset import NematicsDataset
from models import *

preds_dict = {
	'unet': UnetPredictor,
	'r50': fcn_resnet50,
	'r101': fcn_resnet101,
}

if __name__=='__main__':
	parser = argparse.ArgumentParser()

	#Training parameters
	parser.add_argument('-d', '--directory', type=str, default='data')
	parser.add_argument('-b', '--batch_size', type=int, default=2)
	parser.add_argument('--validation_split', type=float, default=0.2)
	parser.add_argument('--num_workers', type=int, default=1)
	parser.add_argument('--crop_size', type=int, default=64)

	#NN parameters
	model_parser = parser.add_argument_group('Model')
	model_parser.add_argument('-p', '--predictor', choices=preds_dict.keys(), default='unet')
	model_parser.add_argument('-c', '--channels', type=int, nargs='+', default=[2,4,6])
	model_parser.add_argument('-r', '--repeat', type=int, default=2)
	model_parser.add_argument('--learning_rate', type=float, default=1e-3)
	model_parser.add_argument('--scheduler_step', type=float, default=0.92)
	parser = pl.Trainer.add_argparse_args(parser)
	args = parser.parse_args()

	# Model
	model = preds_dict[args.predictor](**vars(args))
	print(model.name)

	# Dataset
	dataset = NematicsDataset(args.directory,
		validation_split=args.validation_split,
		transform=model.get_transform(crop_size=args.crop_size))

	
	train_model(model, dataset, args)
