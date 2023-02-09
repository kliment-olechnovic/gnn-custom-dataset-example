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

