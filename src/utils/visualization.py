"""
Visualization utility functions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_feature_importance(
    model_pipeline: Any,
    feature_names: List[str],
    top_n: int = 20,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model_pipeline: Trained model pipeline
        feature_names: List of feature names
        top_n: Number of top features to display
        output_path: Path to save the plot
    """
    try:
        # Extract the model from the pipeline
        model = model_pipeline.named_steps['model']
        
        # Check if the model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            # Get feature importances
            importances = model.feature_importances_
            
            # Create a dataframe of feature importances
            feature_imp = pd.DataFrame(
                {'feature': feature_names, 'importance': importances}
            )
            
            # Sort by importance
            feature_imp = feature_imp.sort_values('importance', ascending=False).head(top_n)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_imp)
            plt.title('Feature Importance')
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Feature importance plot saved to {output_path}")
            else:
                plt.show()
        else:
            logger.warning("Model does not have feature_importances_ attribute")
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")

def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
    """
    try:
        from sklearn.metrics import confusion_matrix
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['No No-Show', 'No-Show'],
            yticklabels=['No No-Show', 'No-Show']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Confusion matrix plot saved to {output_path}")
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")

def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
    """
    try:
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"ROC curve plot saved to {output_path}")
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")

def plot_model_comparison(
    model_metrics: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str],
    output_path: Optional[Path] = None
) -> None:
    """
    Plot comparison of models based on specified metrics.
    
    Args:
        model_metrics: Dictionary with model names as keys and metric dictionaries as values
        metrics_to_plot: List of metrics to include in the plot
        output_path: Path to save the plot
    """
    try:
        # Prepare data for plotting
        data = []
        for model_name, metrics in model_metrics.items():
            for metric in metrics_to_plot:
                if metric in metrics:
                    data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Value': metrics[metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        g = sns.barplot(x='Model', y='Value', hue='Metric', data=df)
        plt.title('Model Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(title='Metric', loc='upper right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Model comparison plot saved to {output_path}")
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error plotting model comparison: {str(e)}") 