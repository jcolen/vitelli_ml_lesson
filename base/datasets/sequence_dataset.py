import os
import re
import glob
import numpy as np
import pandas as pd
import torch
from base_datasets.base_dataset import BaseDataset

class SequenceDataset(BaseDataset):
	def __init__(self, 
				root_dir, 
				transform=None,
				validation_split=0.2):
		super(SequenceDataset, self).__init__(root_dir, transform=transform, validation_split=validation_split)
		
		self.dataframe = self.dataframe.sort_values(['folder', 'idx']).reset_index(drop=True)
		self.next = self.dataframe.loc[(self.dataframe.index + 1) % len(self.dataframe)].reset_index(drop=True)
		mask = self.dataframe.folder == self.next.folder
		self.dataframe = self.dataframe[mask].reset_index(drop=True)
		self.next = self.next[mask].reset_index(drop=True)
		self.split_indices(validation_split)
