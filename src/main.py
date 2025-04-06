"""
Main script to run the hotel no-show prediction pipeline.
"""

import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import gc
from typing import Dict, List, Any

# Add src directory to path to allow imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import from other modules
from config import (
    ROOT_DIR,
    DB_PATH,
    TABLE_NAME,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MODELS,
    METRICS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from data.data_loader import load_data_from_sqlite, split_data
from features.preprocessor import (
    clean_dataset,
    extract_additional_features,
    create_preprocessing_pipeline,
)
from models.model_trainer import (
    create_model_pipeline,
    train_model,
    evaluate_model,
    save_model,
)
from utils.visualization import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_comparison,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT_DIR / "pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hotel No-Show Prediction Pipeline")

    parser.add_argument(
        "--db_path", type=str, default=None, help="Path to SQLite database file"
    )
    parser.add_argument(
        "--table_name", type=str, default=None, help="Name of the table in the database"
    )
    parser.add_argument(
        "--config_file", type=str, default=None, help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=ROOT_DIR / "models",
        help="Directory to save model outputs",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--random_state", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "logistic_regression"
        ],  # Default to just logistic regression to save memory
        help="Models to train",
    )
    parser.add_argument(
        "--no_hyperparameter_tuning",
        action="store_true",
        help="Disable hyperparameter tuning",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=ROOT_DIR / "plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to use (for memory-constrained environments)",
    )

    return parser.parse_args()


def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}


def get_feature_names_from_pipeline(pipeline, categorical_features, numerical_features):
    """Extract feature names from the pipeline's column transformer."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]

        # Extract one-hot encoded feature names
        categorical_encoder = preprocessor.named_transformers_[
            "categorical"
        ].named_steps["encoder"]
        categorical_feature_names = []

        if hasattr(categorical_encoder, "get_feature_names_out"):
            categorical_feature_names = categorical_encoder.get_feature_names_out(
                categorical_features
            ).tolist()

        # Combine with numerical feature names
        feature_names = categorical_feature_names + numerical_features

        return feature_names
    except Exception as e:
        logger.error(f"Error getting feature names: {str(e)}")
        # Return a list of placeholder names if the actual names can't be retrieved
        return [
            f"feature_{i}"
            for i in range(len(categorical_features) * 5 + len(numerical_features))
        ]


def run_pipeline(args):
    """Run the machine learning pipeline."""
    # Process arguments
    db_path = args.db_path or DB_PATH
    table_name = args.table_name or TABLE_NAME
    test_size = args.test_size or TEST_SIZE
    random_state = args.random_state or RANDOM_STATE
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load custom config if provided
    if args.config_file:
        custom_config = load_config_from_file(args.config_file)
        # Update configuration based on custom config
        # (implement as needed based on config structure)

    # Load the data with optional sampling to reduce memory usage
    logger.info("Loading data...")
    if args.sample_size:
        # Use a custom SQL query to sample data
        sample_query = (
            f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {args.sample_size}"
        )
        df = load_data_from_sqlite(db_path, table_name, query=sample_query)
        logger.info(
            f"Using a sample of {args.sample_size} records to reduce memory usage"
        )
    else:
        df = load_data_from_sqlite(db_path, table_name)

    # Clean and preprocess data
    logger.info("Cleaning and preprocessing data...")
    df = clean_dataset(df)
    df = extract_additional_features(df)

    # Free up memory
    gc.collect()

    # Split data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        df, TARGET_COLUMN, test_size=test_size, random_state=random_state
    )

    # Free up memory after splitting
    del df
    gc.collect()

    # Create preprocessing pipeline
    logger.info("Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(
        CATEGORICAL_FEATURES, NUMERICAL_FEATURES
    )

    # Dictionary to store metrics for all models
    all_metrics = {}
    best_model = None
    best_model_name = None
    best_f1_score = 0

    # Train and evaluate models
    for model_name in args.models:
        if model_name not in MODELS:
            logger.warning(f"Unknown model: {model_name}, skipping...")
            continue

        logger.info(f"Processing model: {model_name}")

        model_config = MODELS[model_name]
        model_type = model_config["model_type"]

        # Create model pipeline
        pipeline = create_model_pipeline(preprocessor, model_type)

        # Train model with or without hyperparameter tuning
        if args.no_hyperparameter_tuning:
            trained_model, _ = train_model(pipeline, X_train, y_train)
            params = {}
        else:
            # For memory-intensive operations, use only a subset of parameters
            if args.sample_size:
                # Use simplified parameter grid for sampled data
                simplified_params = {}
                for param, values in model_config["params"].items():
                    simplified_params[param] = (
                        [values[0]] if isinstance(values, list) and values else values
                    )
                trained_model, params = train_model(
                    pipeline, X_train, y_train, param_grid=simplified_params
                )
            else:
                trained_model, params = train_model(
                    pipeline, X_train, y_train, param_grid=model_config["params"]
                )

        # Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test, METRICS)
        all_metrics[model_name] = metrics

        # Save model
        save_model(
            trained_model, model_name, output_dir, params=params, metrics=metrics
        )

        # Create visualizations
        y_pred = trained_model.predict(X_test)
        y_pred_proba = trained_model.predict_proba(X_test)[:, 1]

        # Confusion matrix
        plot_confusion_matrix(
            y_test, y_pred, output_path=plots_dir / f"{model_name}_confusion_matrix.png"
        )

        # ROC curve
        plot_roc_curve(
            y_test, y_pred_proba, output_path=plots_dir / f"{model_name}_roc_curve.png"
        )

        # Feature importance (for tree-based models)
        if model_type in ["random_forest", "gradient_boosting"]:
            try:
                feature_names = get_feature_names_from_pipeline(
                    trained_model, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
                )
                plot_feature_importance(
                    trained_model,
                    feature_names,
                    output_path=plots_dir / f"{model_name}_feature_importance.png",
                )
            except Exception as e:
                logger.error(f"Error plotting feature importance: {str(e)}")

        # Track the best model
        if metrics["f1"] > best_f1_score:
            best_f1_score = metrics["f1"]
            best_model_name = model_name

        # Free up memory after each model
        del trained_model
        gc.collect()

    # Model comparison plot - only if more than one model was trained
    if len(args.models) > 1 and len(all_metrics) > 1:
        plot_model_comparison(
            all_metrics,
            ["accuracy", "precision", "recall", "f1", "roc_auc"],
            output_path=plots_dir / "model_comparison.png",
        )

    # Log best model
    if best_model_name:
        logger.info(f"Best model: {best_model_name} with F1 score: {best_f1_score:.4f}")

    # Save combined metrics
    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    logger.info("Pipeline completed successfully!")

    return all_metrics


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
