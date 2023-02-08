# A real-world example of creating custom datasets in PyTorch Geometric

## Motivation

This is a node-prediction GNN training/evaluation example with a very small amount of data, intended purely to demonstrate how to make a graph dataset for PyTorch Geometric from graph vertices and edges stored in CSV files.

## Usage

Main usage is to read all the "*.py" scripts.

The scripts can also be executed, for eaxmple:

	# train GNN
    python run_training.py
    
    # test a trained GNN model saved after epoch 5
    ./run_evaluation.py testing_data ./output_saved_trained_models/epoch5.pth
    
    # test multiple saved trained GNN models
    find "./output_saved_trained_models/" -type f | sort -V | xargs python ./run_evaluation.py testing_data

## System requirements

Example of installing prerequisites:

    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh
    bash Miniconda3-py39_4.11.0-Linux-x86_64.sh
    
    source ~/miniconda3/bin/activate
    
    conda install pytorch -c pytorch
    conda install pyg -c pyg

