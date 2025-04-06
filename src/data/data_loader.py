"""
Data loader module to fetch data from the SQLite database.
"""
import sqlite3
import pandas as pd
from typing import Tuple, Optional
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data_from_sqlite(
    db_path: str,
    table_name: str,
    query: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from SQLite database into a pandas DataFrame.
    
    Args:
        db_path: Path to the SQLite database
        table_name: Name of the table to query
        query: Custom SQL query (optional)
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        
        if query:
            df = pd.read_sql_query(query, conn)
        else:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            
        conn.close()
        logger.info(f"Loaded {len(df)} records from {table_name}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target variable column
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Check if stratification is possible
    # Stratification requires at least 2 samples for each class
    unique_values = np.unique(y)
    value_counts = y.value_counts()
    min_count = value_counts.min() if not value_counts.empty else 0
    
    if len(unique_values) < 2 or min_count < 2:
        logger.warning("Cannot use stratification: target has insufficient representation. Using random split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    
    logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
    
    return X_train, X_test, y_train, y_test 