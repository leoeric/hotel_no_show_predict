# Hotel No-Show Prediction ML Pipeline

**Name:** Wang Yong  
**Email:** leoeric13@gmail.com

## Overview

This project implements an end-to-end machine learning pipeline for predicting hotel customer no-shows. The pipeline is designed to help a hotel chain formulate policies to reduce expenses incurred due to no-shows by identifying customers who are likely to not show up for their bookings.

## Folder Structure

```
hotel_no_show_predict/
│
├── db/                    # Database directory
│   └── noshow.db          # SQLite database file
│
├── src/                   # Source code directory
│   ├── __init__.py
│   ├── config.py          # Configuration parameters
│   ├── main.py            # Main script to run the pipeline
│   │
│   ├── data/              # Data loading modules
│   │   ├── __init__.py
│   │   └── data_loader.py # Functions to load data from SQLite
│   │
│   ├── features/          # Feature engineering modules
│   │   ├── __init__.py
│   │   └── preprocessor.py # Feature preprocessing and transformation
│   │
│   ├── models/            # Model training modules
│   │   ├── __init__.py
│   │   └── model_trainer.py # Model training and evaluation
│   │
│   └── utils/             # Utility modules
│       ├── __init__.py
│       └── visualization.py # Visualization utilities
│
├── models/                # Directory for saved models (created at runtime)
├── plots/                 # Directory for saved plots (created at runtime)
├── pipeline.log           # Log file (created at runtime)
├── requirements.txt       # Project dependencies
├── run.sh                 # Executable bash script to run the pipeline
└── README.md              # This file
```

## Pipeline Execution Instructions

### Prerequisites

- Python 3.8 or higher
- SQLite3

### Running the Pipeline

To run the pipeline with default configuration:

```bash
./run.sh
```

This will execute the pipeline with pre-configured settings.

### Customizing Pipeline Parameters

You can modify pipeline parameters in several ways:

1. **Command-line Arguments**:
   ```bash
   python src/main.py --models logistic_regression random_forest --output_dir ./custom_models --no_hyperparameter_tuning
   ```

   Available command-line options:
   - `--db_path`: Path to the SQLite database file
   - `--table_name`: Name of the table in the database
   - `--config_file`: Path to a custom configuration JSON file
   - `--output_dir`: Directory to save model outputs
   - `--test_size`: Proportion of data to use for testing
   - `--random_state`: Random seed for reproducibility
   - `--models`: Models to train (space-separated list)
   - `--no_hyperparameter_tuning`: Disable hyperparameter tuning
   - `--plots_dir`: Directory to save plots

2. **Configuration File** (`src/config.py`):
   - You can modify the default configuration parameters in this file.
   - Parameters include data paths, feature lists, model hyperparameters, etc.

3. **Custom Configuration File**:
   - Create a JSON configuration file and provide it with the `--config_file` argument.

## Pipeline Flow

The machine learning pipeline follows these logical steps:

1. **Data Loading**: 
   - Loads hotel booking data from the SQLite database.

2. **Data Cleaning and Preprocessing**:
   - Handles missing values
   - Converts data types
   - Extracts additional features

3. **Feature Engineering**:
   - Converts categorical features using one-hot encoding
   - Scales numerical features
   - Creates derived features like stay duration and lead time

4. **Model Training**:
   - Trains multiple machine learning models
   - Performs hyperparameter tuning (optional)
   - Evaluates models using various metrics

5. **Model Evaluation and Visualization**:
   - Generates performance metrics
   - Creates visualizations (confusion matrices, ROC curves, etc.)
   - Compares models

6. **Results Storage**:
   - Saves trained models
   - Stores evaluation metrics
   - Saves visualizations

![Pipeline Flow Diagram](https://mermaid.ink/img/pako:eNqNksFqwzAMhl_F-NRCXwDvsPZQaClkl-6g2UqjENsltme69L13kkIHg7GdJP3fp19yjryxBnni0LSuiE7pDW4CHnD04O2AbnUdmFD2xq3vdccpLF2PDXq_-1vEFGazeeY0g9mFvsGuiY2RZOYPKCl7qyNK4q3gJdSGihsUSJKFDRXN3oSQm2wkYtKu6mCvPVLIf2gLAo2Vr0gOPHNxAZPcWSMQQ4FkZoXYKnHJRhlvsBL5rEcXOhqw_RNdbNSQzZ-uR_4-kEu78bBTGwV3JhfoKxTz3OraPWy1kbvYXsvIvPE0Yd_jTZfAXXk9DG6o4JnfS9f7lzQuZeKF1xUuKDu-8Ct8bfnKX7nQdOKKX3jDKyWb-Be2OaxZ)

## EDA Summary and Feature Engineering

Analysis of the dataset revealed several key insights that influenced the pipeline design:

### Key Findings from EDA

1. **Class Distribution**: The dataset has an imbalanced class distribution with approximately 63% non-no-shows and 37% no-shows.

2. **Missing Values**: Several columns contained missing values, particularly in price, num_adults, and num_children fields.

3. **Temporal Patterns**: Bookings made further in advance (longer lead time) showed higher no-show rates.

4. **Price and Currency**: Price values varied significantly and were stored with different currency indicators (SGD$, USD$).

5. **Booking Source**: Bookings made through certain platforms showed different no-show patterns.

### Feature Processing Summary

| Feature | Type | Transformation | Notes |
|---------|------|----------------|-------|
| booking_id | Numerical | Dropped | Identifier, not useful for prediction |
| no_show | Numerical | Target | Binary target variable (0/1) |
| branch | Categorical | One-hot encoded | Hotel branch location |
| booking_month | Categorical | One-hot encoded + Numeric conversion | Month of booking |
| arrival_month | Categorical | One-hot encoded + Numeric conversion | Month of arrival |
| arrival_day | Numerical | Scaled | Day of arrival |
| checkout_month | Categorical | One-hot encoded + Numeric conversion | Month of checkout |
| checkout_day | Numerical | Scaled | Day of checkout |
| country | Categorical | One-hot encoded | Origin country of guest |
| first_time | Categorical | One-hot encoded + Binary conversion | Whether guest is first-time customer |
| room | Categorical | One-hot encoded | Room type |
| price | Categorical | Currency extraction + Numeric value extraction | Price was split into currency type and numeric value |
| platform | Categorical | One-hot encoded | Booking platform |
| num_adults | Numerical | Imputed with median + Scaled | Number of adults |
| num_children | Numerical | Imputed with 0 + Scaled | Number of children |

### Derived Features

1. **stay_duration**: Duration of stay calculated from checkout and arrival dates
2. **lead_time_months**: Time between booking and arrival in months
3. **currency**: Extracted currency type from price
4. **price_value**: Extracted numeric value from price
5. **is_first_time**: Binary indicator for first-time guests
6. **total_occupants**: Sum of adults and children

## Model Selection and Evaluation

Three different models were evaluated for this task:

1. **Logistic Regression**:
   - A baseline linear model for binary classification
   - Advantages: Interpretable, fast to train, provides probability estimates
   - Disadvantages: May not capture complex non-linear relationships

2. **Random Forest**:
   - An ensemble of decision trees
   - Advantages: Handles non-linear relationships, robust to outliers, provides feature importance
   - Disadvantages: Less interpretable than logistic regression, more parameters to tune

3. **Gradient Boosting**:
   - Sequential ensemble method that builds trees to correct errors of previous trees
   - Advantages: Often achieves state-of-the-art performance, handles various data types well
   - Disadvantages: Can overfit, more complex to tune, slower training

### Evaluation Metrics

The models were evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive identifications that were actually correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve

F1 score was prioritized for model selection as it balances precision and recall, which is important for the no-show prediction task where both false positives and false negatives have business implications.

## Conclusion

The pipeline provides a flexible framework for training and evaluating models to predict hotel no-shows. By identifying customers who are likely to not show up, hotels can implement targeted strategies to reduce associated costs, such as overbooking policies, confirmation reminders, or special incentives.

The modular design allows for easy experimentation with different features, preprocessing techniques, and models to improve prediction performance.

### Memory Optimization Options

For environments with limited memory, the pipeline provides options to reduce memory usage:

1. **Data Sampling**:
   ```bash
   python src/main.py --sample_size 50000
   ```
   This limits the number of records used for training, which significantly reduces memory requirements.

2. **Single Model Execution**:
   ```bash
   python src/main.py --models logistic_regression
   ```
   Training only one model at a time reduces peak memory usage.

3. **Hyperparameter Tuning Control**:
   ```bash
   python src/main.py --no_hyperparameter_tuning
   ```
   Disables grid search which can be memory-intensive.

The default `run.sh` script is configured to use memory optimization options to ensure the pipeline runs successfully on most systems.