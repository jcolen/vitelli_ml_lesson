import os
import re
import glob
import torch
import numpy as np

from base.datasets.base_dataset import BaseDataset
from winding import winding

import warnings
warnings.filterwarnings("ignore")

'''
Dataset to hold active nematics information
'''
class NematicsDataset(BaseDataset):
	def list_file_indices(self, path):
		idxs = None
		fnames = glob.glob(os.path.join(path, 'nx*'))
		inds = [list(map(int, re.findall(r'\d+', os.path.basename(fname))))[-1] for fname in fnames]
		idxs = inds if idxs is None else np.intersect1d(idxs, inds)
		return np.sort(idxs)

	def get_image(self, idx):
		subdir = os.path.join(self.root_dir, self.dataframe.folder[idx])
		ind = self.dataframe.idx[idx]
		nx = np.loadtxt(os.path.join(subdir, 'nx%d' % ind))
		ny = np.loadtxt(os.path.join(subdir, 'ny%d' % ind))
		return {'nx': nx, 'ny': ny}
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		sample = self.get_image(idx)
		subdir = os.path.join(self.root_dir, self.dataframe.folder[idx])
		ind = self.dataframe.idx[idx]
		sample = {
			'nx': np.loadtxt(os.path.join(subdir, 'nx%d' % ind)),
			'ny': np.loadtxt(os.path.join(subdir, 'ny%d' % ind)),
		}

		if self.transform:
			sample = self.transform(sample)

		return sample

'''
Additional data processing
'''
class ProcessInputs(object):
	def __call__(self, sample):
		nx, ny = sample['nx'], sample['ny']
		cos_squared = nx * nx
		sin2t = 2 * nx * ny
		cos2t = nx * nx - ny * ny
		theta = np.arctan2(ny, nx)
		theta[theta < 0] += np.pi
		theta[theta > np.pi] -= np.pi
		wind = winding(theta, radius=2)
		label = (2 * wind + 1).astype(int)
		return {'fluorescence': cos_squared[None],
				'sincos': np.stack((sin2t, cos2t)),
				'theta': theta[None],
				'winding': wind[None],
				'label': label}
