import sys
import pandas
import torch
import torch_geometric
import numpy

def read_graph(vertices_file, edges_file):
	df_vertices_in=pandas.read_csv(vertices_file, index_col="id")
	df_edges=pandas.read_csv(edges_file)
	x=torch.tensor(df_vertices_in[['area', 'boundary', 'adjacency', 'VE', 'VESSa', 'VESSb', 'MVE', 'MVESSa', 'MVESSb', 'VE_norm', 'VESSa_norm', 'VESSb_norm', 'MVE_norm', 'MVESSa_norm', 'MVESSb_norm']].values, dtype=torch.float32)
	edge_index=torch.tensor(df_edges[['from', 'to']].values.T, dtype=torch.long)
	edge_attr=torch.tensor(df_edges[['weight']].values, dtype=torch.float32)
	graph=torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
	return graph

model_file=sys.argv[1]
input_vertices_file=sys.argv[2]
input_edges_file=sys.argv[3]
output_vertices_predictions_file=sys.argv[4]

device=torch.device('cpu')

data=read_graph(input_vertices_file, input_edges_file)
data=data.to(device)

model=torch.load(model_file)
model=model.to(device)
model.eval()

pred_y=model(data)

numpy.savetxt(output_vertices_predictions_file, pred_y.squeeze().detach().numpy(), fmt='%10.5f')
