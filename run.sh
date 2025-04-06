#!/bin/bash

# Print error message and exit
function error_exit {
    echo "$1" 1>&2
    exit 1
}

echo "Starting hotel no-show prediction pipeline..."

# Check if python is available
command -v python > /dev/null 2>&1 || error_exit "Python is required but not installed or not in PATH. Aborting."

# Run the main script with error handling and memory optimization
# Using a sample size of 50,000 records to avoid memory issues
# Only running logistic regression model to reduce memory usage
python src/main.py --no_hyperparameter_tuning --sample_size 50000 --models logistic_regression || error_exit "Pipeline execution failed!"

echo "Pipeline completed successfully!"

echo "To run with all data and models (requires more memory):"
echo "python src/main.py --no_hyperparameter_tuning --models logistic_regression random_forest gradient_boosting" 