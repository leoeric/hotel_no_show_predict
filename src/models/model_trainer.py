"""
Model training and evaluation module.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def get_model(model_type: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a model instance based on model type.
    
    Args:
        model_type: Type of model to create ('logistic_regression', 'random_forest', or 'gradient_boosting')
        params: Model parameters
        
    Returns:
        Model instance
    """
    if params is None:
        params = {}
        
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_model_pipeline(
    preprocessor: ColumnTransformer,
    model_type: str,
    model_params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Create a full pipeline with preprocessing and model.
    
    Args:
        preprocessor: Feature preprocessing pipeline
        model_type: Type of model to use
        model_params: Model parameters
        
    Returns:
        Pipeline with preprocessing and model
    """
    model = get_model(model_type, model_params)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    logger.info(f"Created model pipeline with {model_type}")
    
    return pipeline

def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv: int = 5
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a model with optional hyperparameter tuning.
    
    Args:
        pipeline: Model pipeline
        X_train: Training features
        y_train: Training target
        param_grid: Grid of parameters to search
        cv: Number of cross-validation folds
        
    Returns:
        Trained pipeline and best parameters
    """
    if param_grid:
        grid_params = {}
        for param, values in param_grid.items():
            grid_params[f'model__{param}'] = values
            
        grid_search = GridSearchCV(
            pipeline,
            param_grid=grid_params,
            cv=cv,
            scoring='f1',
            verbose=1,
            n_jobs=-1
        )
        
        logger.info("Starting grid search...")
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    else:
        logger.info("Training model...")
        pipeline.fit(X_train, y_train)
        return pipeline, {}

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: List[str]
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model pipeline
        X_test: Testing features
        y_test: Testing target
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results = {}
    
    for metric in metrics:
        if metric == 'accuracy':
            results[metric] = accuracy_score(y_test, y_pred)
        elif metric == 'precision':
            results[metric] = precision_score(y_test, y_pred)
        elif metric == 'recall':
            results[metric] = recall_score(y_test, y_pred)
        elif metric == 'f1':
            results[metric] = f1_score(y_test, y_pred)
        elif metric == 'roc_auc' and y_pred_proba is not None:
            results[metric] = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Evaluation results: {results}")
    
    return results

def save_model(
    model: Pipeline,
    model_name: str,
    output_dir: Path,
    params: Dict[str, Any] = None,
    metrics: Dict[str, float] = None
) -> None:
    """
    Save the trained model and its metadata.
    
    Args:
        model: Trained model pipeline
        model_name: Name of the model
        output_dir: Directory to save the model
        params: Model parameters
        metrics: Model evaluation metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "params": params or {},
        "metrics": metrics or {},
        "feature_names": model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None
    }
    
    metadata_path = output_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}") 