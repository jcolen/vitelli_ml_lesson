import os
import pandas as pd
import numpy as np
import random
import scipy 
import skimage.transform

import torch
from torch.utils.data import Dataset
try:
	from torch.utils.data.sampler import SubsetRandomSampler
except:
	from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

'''
Generic abstract Dataset class with some predefined methods/structures
'''
class BaseDataset(Dataset):
	'''
	Arguments:
		root_dir(string) - The directory to look for files
		transform(callable, optional) - Optional transform to be applied to a sample
		force_load - Require reloading of dataset ignoring existing csv file
		validation_split - train/validation split
	'''
	def __init__(self, 
				 root_dir, 
				 transform=None,
				 force_load=True,
				 validation_split=0.2):
		self.root_dir = root_dir
		self.transform = transform

		index_name = os.path.join(root_dir, 'index.csv')
		if not force_load and os.path.exists(index_name):
			self.dataframe = pd.read_csv(index_name)
		else:
			self.dataframe = self.build_files_index()
		
		self.dataframe.to_csv(index_name, index=False)
		self.folders = self.dataframe.folder.unique()
		self.num_folders = len(self.folders)
		print('Found %d items in %d folders' % (len(self), self.num_folders))
		self.split_indices(validation_split)

	def __len__(self):
		return len(self.dataframe)
	
	def split_indices(self, validation_split):
		split	= int(np.floor(validation_split * len(self)))

		nfolders = len(self.folders)
		fidxs = np.arange(nfolders)
		np.random.shuffle(fidxs)
		vidxs = int(np.ceil(validation_split * nfolders))
		vfolders = self.folders[fidxs[:vidxs]]
		tfolders = self.folders[fidxs[vidxs:]]

		self.train_indices = self.dataframe[self.dataframe.folder.isin(tfolders)].index.to_numpy()
		self.validation_indices = self.dataframe[self.dataframe.folder.isin(vfolders)].index.to_numpy()

		print('Training set includes %d folders and %d items' % (nfolders-vidxs, len(self.train_indices)))
		print('Validation set includes %d folders and %d items' % (vidxs, len(self.validation_indices)))

	def get_loader(self, indices, batch_size, num_workers, pin_memory=True):
		sampler = SubsetRandomSampler(indices)
		loader = DataLoader(self, 
			batch_size=batch_size,
			num_workers=num_workers,
			sampler=sampler,
			pin_memory=pin_memory)
		return loader
	
	def take_folder(self, folder, validation_split=0.2):
		self.dataframe = self.dataframe[self.dataframe.folder == folder]
		self.dataframe = self.dataframe.reset_index(drop=True)
		self.split_indices(validation_split)
		print('Dataset now has %d items' % len(self))

	#Build an index of the images in different class subfolders
	def build_files_index(self):
		dataframe = pd.DataFrame(columns=['folder', 'idx'])
		folders, idxs = [], np.zeros(0, dtype=int)
		for subdir in os.listdir(self.root_dir):
			dirpath = os.path.join(self.root_dir, subdir)
			if not os.path.isdir(dirpath):
				continue
			
			inds = self.list_file_indices(dirpath)
			if inds is None:
				continue

			dfol = pd.DataFrame({'folder': [subdir,] * len(inds), 'idx': inds})
			dataframe = dataframe.append(dfol, ignore_index=True)

		return dataframe
	
	def list_file_indices(self, path):
		pass

	def __getitem__(self, idx):
		pass


'''
Basic data processing transforms
'''

class RandomCrop(object):
	def __init__(self, crop_size, ndims=2, skip=[]):
		self.ndims = ndims
		self.skip = skip
		
		assert isinstance(crop_size, (int, tuple))
		if isinstance(crop_size, int):
			self.crop_size = (crop_size, ) * self.ndims
		else:
			assert len(crop_size) == self.ndims
			self.crop_size = crop_size

	def __call__(self, sample):
		x = sample[next(iter(sample))]
		
		dims = x.shape[-self.ndims:]
		corner = [np.random.randint(0, d-nd) for d, nd in zip(dims, self.crop_size)]

		crop_indices = tuple(np.s_[c:c+nd] for c, nd in zip(corner, self.crop_size))
		for key in sample.keys():
			if key in self.skip:	continue
			same_indices = tuple(np.s_[0:d] for d in sample[key].shape[:-self.ndims])
			indices = same_indices + crop_indices
			sample[key] = sample[key][indices]
		
		return sample

class RandomTranspose(object):
	def __init__(self, prob=0.5, ndims=2, skip=[]):
		self.prob  = prob
		self.ndims = ndims
		self.skip = skip

	def __call__(self, sample):
		if np.random.random() < self.prob:
			axes = random.sample(range(-self.ndims, 0), 2)
			for key in sample.keys():
				if key in self.skip:	continue
				sample[key] = np.swapaxes(sample[key], axes[0], axes[1])
			
		return sample
	
class RandomFlip(object):
	def __init__(self, prob=0.5, ndims=2, skip=[]):
		self.prob  = prob
		self.ndims = ndims
		self.skip = skip

	def __call__(self, sample):
		for dim in range(-self.ndims, 0):
			if np.random.random() < self.prob:
				for key in sample.keys():
					if key in self.skip:	continue
					sample[key] = np.flip(sample[key], axis=dim)
				
		return sample

class ToTensor(object):
	def __call__(self, sample):
		for key in sample.keys():
			sample[key] = torch.tensor(sample[key].copy(), 
				dtype=torch.int64 if sample[key].dtype == int else torch.float32)
		return sample
