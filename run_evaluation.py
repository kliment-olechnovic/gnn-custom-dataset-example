import sys
import torch

import custom_dataset_from_graph_csv_files

dataset_name=sys.argv[1]
model_files=sys.argv[2:]

dataset=custom_dataset_from_graph_csv_files.Dataset(root=dataset_name)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def calc_mse(data):
	model.eval()
	data=data.to(device)
	pred_y=model(data)
	loss=torch.nn.functional.mse_loss(pred_y.squeeze(), data.y.squeeze())
	return loss.item()

for model_file in model_files:
	model=torch.load(model_file)
	model=model.to(device)
	
	mse_sum=0
	for data in dataset:
		mse_sum+=calc_mse(data)
	
	print('model =', model_file, ' mse =', mse_sum/len(dataset))

