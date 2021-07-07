import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d

class CnnCell(nn.Module):
	def __init__(self, in_channel, out_channel, dropout=0.1):
		super(CnnCell, self).__init__()
		
		self.conv = Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
		self.bn   = nn.BatchNorm2d(out_channel)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = torch.tanh(x)
		x = self.dropout(x)
		return x

class DeCnnCell(nn.Module):
	def __init__(self, in_channel, out_channel, dropout=0.1):
		super(DeCnnCell, self).__init__()
		
		self.deconv = ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
		self.bn		= nn.BatchNorm2d(out_channel)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.deconv(x)
		x = self.bn(x)
		x = torch.tanh(x)
		x = self.dropout(x)
		return x

'''
Size-preserving CNN cell
'''
class FlatCnnCell(nn.Module):
	def __init__(self, in_channel, out_channel, depth=1, residual=True):
		super(FlatCnnCell, self).__init__()
		
		self.conv = Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
		self.bn   = nn.BatchNorm2d(out_channel)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = torch.tanh(x)
		return x
	
class RepeatFlatCnnCell(nn.Module):
	def __init__(self, in_channels, out_channels, repeat=2, residual=False):
		super(RepeatFlatCnnCell, self).__init__()

		self.cells = nn.ModuleList()
		self.residual = residual
		
		#Skip connection with concatentation
		for i in range(repeat-1):
			self.cells.append(FlatCnnCell(in_channels, in_channels))
		self.cells.append(FlatCnnCell(in_channels, out_channels))
	
	def forward(self, x):
		out = self.cells[0](x)
		for cell in self.cells[1:]:
			out = cell(out)
		#Concatenate along channel dimension
		if self.residual:	return torch.cat((out, x), dim=1)
		else:				return out


'''
Interpolating/pooling size-changing cells
'''
class UpsampleCell(nn.Module):
	def __init__(self, in_channel, out_channel, dropout=0.1, method='bilinear'):
		super(UpsampleCell, self).__init__()

		self.upsample = nn.Upsample(scale_factor=2, mode=method, align_corners=False)
		self.conv = FlatCnnCell(in_channel, out_channel)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.conv(x)
		x = self.upsample(x)
		x = self.dropout(x)
		return x

class DownsampleCell(nn.Module):
	def __init__(self, in_channel, out_channel, dropout=0.1):
		super(DownsampleCell, self).__init__()

		self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
		self.conv = FlatCnnCell(in_channel, out_channel)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x = self.conv(x)
		x = self.downsample(x)
		x = self.dropout(x)
		return x
