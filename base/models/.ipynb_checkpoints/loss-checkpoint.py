import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpWeightedLoss(nn.Module):
	def __init__(self, base_loss):
		super(ExpWeightedLoss, self).__init__()
		self.base_loss = base_loss
	
	def forward(self, input, target):
		mag = target.norm(p=3, dim=1, keepdim=True)
		base = self.base_loss(input, target, reduction='none')
		return torch.mean(base * mag.exp())

class ResidualLoss(nn.Module):
	def forward(self, u, v):
		uavg = torch.sqrt((u**2).sum(dim=1).mean(dim=(-1,-2)))
		vavg = torch.sqrt((v**2).sum(dim=1).mean(dim=(-1,-2)))
		residual = uavg[:, None, None]**2 * torch.einsum('ijkl,ijkl->ikl', v, v)
		residual += vavg[:, None, None]**2 * torch.einsum('ijkl,ijkl->ikl', u, u)
		residual -= 2 * vavg[:, None, None] * uavg[:, None, None] * torch.einsum('ijkl,ijkl->ikl', v, u)
		residual /= 2 * vavg[:, None, None]**2 * uavg[:, None, None]**2
		return residual.mean()

class BetaVAELoss(nn.Module):
	def __init__(self, beta=0.0001, base_loss=F.mse_loss):
		super(BetaVAELoss, self).__init__()
		self.beta = beta
		self.base_loss = base_loss
	
	def forward(self, input, target):
		pred, param, logvar = input
		yloss = self.base_loss(pred, target)
		vaeloss = 0.5 * torch.mean(torch.sum(param*param + logvar.exp() - logvar - 1, dim=-1))
		return yloss + self.beta * vaeloss

class JointLoss(nn.Module):
	def __init__(self, base_loss):
		super(JointLoss, self).__init__()
		self.base_loss = base_loss
	
	def forward(self, input, target):
		return self.base_loss(input[0], target[0]) + \
			   self.base_loss(input[1], target[1])
