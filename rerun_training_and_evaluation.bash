#!/bin/bash

cd "$(dirname $0)"

echo "Clearing previously generated files"

rm -rf ./training_data ./validation_data ./testing_data ./output_saved_trained_models ./__pycache__

echo "Running training using 'training_data'"

python run_training.py

echo "Running evaluation using 'training_data'"

find "./output_saved_trained_models/" -type f | sort -V | xargs python ./run_evaluation.py training_data

echo "Running evaluation using 'validation_data'"

find "./output_saved_trained_models/" -type f | sort -V | xargs python ./run_evaluation.py validation_data

echo "Running evaluation using 'testing_data'"

find "./output_saved_trained_models/" -type f | sort -V | xargs python ./run_evaluation.py testing_data

