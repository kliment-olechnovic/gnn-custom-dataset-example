import os
import torch
import torch_geometric

import custom_dataset_from_graph_csv_files
import custom_gnn_model

dataset=custom_dataset_from_graph_csv_files.CustomDataset(root='training_data')
dataset.shuffle()

data_loader=torch_geometric.loader.DataLoader(dataset, batch_size=4)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=custom_gnn_model.CustomGNN()
model=model.to(device)

optimizer=torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

def train():
	model.train()
	loss_sum=0
	for data in data_loader:
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

number_of_epochs=25;
saving_period=1;

for epoch in range(1, number_of_epochs+1):
	print('epoch =', epoch, ' mse =', train()/len(dataset))
	if (epoch==0) or (epoch%saving_period==0) or (epoch==number_of_epochs):
		torch.save(model, output_directory+'/epoch'+str(epoch)+'.pth')

