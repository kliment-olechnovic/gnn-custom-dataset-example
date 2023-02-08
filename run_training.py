import os
import torch
import torch_geometric

import custom_dataset_from_graph_csv_files
import custom_gnn_model
import custom_optimizer

dataset=custom_dataset_from_graph_csv_files.Dataset(root='training_data')
dataset.shuffle()

train_loader=torch_geometric.loader.DataLoader(dataset, batch_size=32)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=custom_gnn_model.GNN()
model=model.to(device)

optimizer=custom_optimizer.optimizer(model)

def train():
	model.train()
	loss_sum=0
	for data in train_loader:
		data=data.to(device)
		optimizer.zero_grad()
		pred_y=model(data)
		loss=torch.nn.functional.mse_loss(pred_y.squeeze(), data.y.squeeze())
		loss.backward()
		loss_sum+=data.num_graphs*loss.item()
		optimizer.step()
	return loss_sum

output_directory='./output_saved_trained_models'
if not os.path.exists(output_directory):
	os.makedirs(output_directory)

number_of_epochs=15;
saving_period=1;

for epoch in range(1, number_of_epochs+1):
	print(epoch, train()/len(dataset))
	if (epoch==0) or (epoch%saving_period==0) or (epoch==number_of_epochs):
		torch.save(model, output_directory+'/epoch'+str(epoch)+'.pth')

