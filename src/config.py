"""
Configuration file for the hotel no-show prediction pipeline.
This file contains all configurable parameters for the ML pipeline.
"""
import os
from pathlib import Path

# Project structure
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "db"
DB_PATH = DATA_DIR / "noshow.db"
TABLE_NAME = "noshow"

# Data processing
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "no_show"

# Feature engineering
CATEGORICAL_FEATURES = [
    "branch", 
    "booking_month", 
    "arrival_month", 
    "checkout_month", 
    "country", 
    "first_time", 
    "room", 
    "price", 
    "platform"
]

NUMERICAL_FEATURES = [
    "arrival_day", 
    "checkout_day", 
    "num_adults", 
    "num_children"
]

# Models and hyperparameters
MODELS = {
    "logistic_regression": {
        "model_type": "logistic_regression",
        "params": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }
    },
    "random_forest": {
        "model_type": "random_forest",
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "class_weight": ["balanced", None]
        }
    },
    "gradient_boosting": {
        "model_type": "gradient_boosting",
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0]
        }
    }
}

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"] 