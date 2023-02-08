import torch

def optimizer(model):
	return torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
