import pandas
import torch
import torch_geometric


def read_list_of_strings(list_file):
	strings=[]
	with open(list_file) as file:
		strings=[line.rstrip() for line in file]
	return strings


def read_graph(files_prefix):
	data_frame_vertices_in=pandas.read_csv(files_prefix+"_vertices_in.csv", index_col="id")
	data_frame_vertices_out=pandas.read_csv(files_prefix+"_vertices_out.csv", index_col="id")
	data_frame_edges=pandas.read_csv(files_prefix+"_edges.csv")
	
	x=torch.tensor(data_frame_vertices_in[['area', 'boundary', 'adjacency', 'VE', 'VESSa', 'VESSb', 'MVE', 'MVESSa', 'MVESSb', 'VE_norm', 'VESSa_norm', 'VESSb_norm', 'MVE_norm', 'MVESSa_norm', 'MVESSb_norm']].values, dtype=torch.float32)
	y=torch.tensor(data_frame_vertices_out[['gt0_pe1']].values, dtype=torch.float32)
	edge_index=torch.tensor(data_frame_edges[['from', 'to']].values.T, dtype=torch.long)
	edge_attr=torch.tensor(data_frame_edges[['weight']].values, dtype=torch.float32)
	
	graph=torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
	
	return graph


class CustomDataset(torch_geometric.data.InMemoryDataset):
	def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, log=False):
		super().__init__(root, transform, pre_transform, pre_filter, log)
		self.data, self.slices = torch.load(self.processed_paths[0])
	
	@property
	def processed_file_names(self):
		return ['data.pt']
	
	def process(self):
		raw_prefixes=read_list_of_strings("./input_graph_CSV_files/"+self.root+"_files_prefixes.txt")
		data_list=[read_graph("./input_graph_CSV_files/"+raw_prefix) for raw_prefix in raw_prefixes]
	
		if self.pre_filter is not None:
			data_list=[data for data in data_list if self.pre_filter(data)]
	
		if self.pre_transform is not None:
			data_list=[self.pre_transform(data) for data in data_list]
	
		data, slices = self.collate(data_list)
		torch.save((data, slices), self.processed_paths[0])

