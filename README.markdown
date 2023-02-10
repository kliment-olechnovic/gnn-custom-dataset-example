# A real-world example of creating custom datasets in PyTorch Geometric

This repository is intended purely to demonstrate how to make a graph dataset for PyTorch Geometric from graph vertices and edges stored in CSV files.

The demonstration is done through a node-prediction GNN training/evaluation example with a very small amount of code and data.

## Usage

Main usage is to read all the "*.py" scripts.

The scripts can also be executed, for example:

	# train GNN
    python run_training.py
    
    # test a trained GNN model saved after epoch 5
    python run_evaluation.py testing_data ./output_saved_trained_models/epoch5.pth
    
    # test multiple saved trained GNN models
    find "./output_saved_trained_models/" -type f | sort -V | xargs python ./run_evaluation.py testing_data
    
    # use a single trained model to predict vertex values for a single graph, and save the predictions to a file
    python run_inference_for_one_graph.py \
      ./output_saved_trained_models/epoch15.pth \
      ./input_graph_CSV_files/data/1A22/1A22_sr1_vertices_in.csv \
      ./input_graph_CSV_files/data/1A22/1A22_sr1_edges.csv \
      ./output_vertex_predictions.txt

## System requirements

Example of installing prerequisites:

    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
    bash Miniconda3-py39_4.11.0-Linux-x86_64.sh
    
    source ~/miniconda3/bin/activate
    
    conda install pytorch -c pytorch
    conda install pyg -c pyg

## Remarks about data

The included input graphs are already prepared for GNN training and application.
The graph preparation code is not included, but below are the main recommendations for the graphs to work with the provided training and inference code.

### Graph connectivity

The graphs should have bidirectional connections and self-connections.
That is, in any '*_edges.csv' file:

 * if there is (i -> j) edge, there should also be (j -> i) edge with the same weight
 * there should be (i -> i) edge with an apppropriate weight for every vertex id i

### Normalization of vertex and edge feature values

All the vertex and edge feature values should be normalized universally (not on per-graph basis, but based on some global statistics) - for example, converted to z-scores using mean and standard deviation values known beforehand or derived from all the graphs used in training:

    z_score = (x - mean) / standard_deviation

