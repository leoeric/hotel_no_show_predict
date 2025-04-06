"""
Feature preprocessing module for data transformations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

def create_preprocessing_pipeline(
    categorical_features: List[str],
    numerical_features: List[str]
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for categorical and numerical features.
    
    Args:
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        
    Returns:
        ColumnTransformer preprocessing pipeline
    """
    # Categorical features pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Numerical features pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine pipelines into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_pipeline, categorical_features),
            ('numerical', numerical_pipeline, numerical_features)
        ],
        remainder='drop'
    )
    
    logger.info("Created preprocessing pipeline")
    
    return preprocessor

def extract_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract additional features from the existing data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Convert month names to numbers
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    # Time-related features
    for col in ['booking_month', 'arrival_month', 'checkout_month']:
        if col in df.columns:
            # Handle potential missing values - avoid inplace operations
            df[f'{col}_num'] = df[col].map(month_mapping)
            # Fill missing values with the median month number if possible
            if df[f'{col}_num'].count() > 0:
                median_month = df[f'{col}_num'].median()
                df[f'{col}_num'] = df[f'{col}_num'].fillna(median_month)
            else:
                # If all values are missing, use a default value of 6 (middle of the year)
                df[f'{col}_num'] = 6
    
    # Calculate stay duration
    if all(col in df.columns for col in ['arrival_month_num', 'arrival_day', 'checkout_month_num', 'checkout_day']):
        try:
            # Handle year transitions (December to January)
            df['month_diff'] = df['checkout_month_num'] - df['arrival_month_num']
            df['month_diff'] = df['month_diff'].apply(lambda x: x if x >= 0 else x + 12)
            
            df['day_diff'] = df['checkout_day'] - df['arrival_day']
            df['day_diff'] = df.apply(
                lambda row: row['day_diff'] if row['month_diff'] == 0 else 
                row['day_diff'] + 30 * row['month_diff'], axis=1
            )
            
            df['stay_duration'] = df['day_diff']
            # Drop intermediate columns
            df = df.drop(columns=['month_diff', 'day_diff'])
        except Exception as e:
            # Log the error but continue processing
            logger.error(f"Error calculating stay duration: {str(e)}")
    
    # Calculate lead time (days between booking and arrival)
    if all(col in df.columns for col in ['booking_month_num', 'arrival_month_num']):
        try:
            df['lead_time_months'] = df['arrival_month_num'] - df['booking_month_num']
            df['lead_time_months'] = df['lead_time_months'].apply(lambda x: x if x >= 0 else x + 12)
        except Exception as e:
            # Log the error but continue processing
            logger.error(f"Error calculating lead time: {str(e)}")
    
    # Extract currency from price
    if 'price' in df.columns:
        try:
            # Extract currency, handling missing values - avoid inplace
            df['currency'] = df['price'].astype(str).str.extract(r'(\w+)\$')
            df['currency'] = df['currency'].fillna('Unknown')
            
            # Extract numeric price value
            price_values = df['price'].astype(str).str.extract(r'\$\s*(\d+\.?\d*)')
            df['price_value'] = pd.to_numeric(price_values[0], errors='coerce')
            
            # Clean missing values - avoid inplace
            median_price = df['price_value'].median()
            if pd.notna(median_price):
                df['price_value'] = df['price_value'].fillna(median_price)
            else:
                # If all values are missing/NaN, use a default value
                df['price_value'] = df['price_value'].fillna(0)
        except Exception as e:
            # Log the error but continue processing
            logger.error(f"Error processing price: {str(e)}")
            # Create placeholder columns
            df['currency'] = 'Unknown'
            df['price_value'] = 0
    
    # Convert categorical yes/no to numeric
    if 'first_time' in df.columns:
        df['is_first_time'] = df['first_time'].map({'Yes': 1, 'No': 0})
        # Handle any values that weren't mapped - avoid inplace
        df['is_first_time'] = df['is_first_time'].fillna(0)
    
    # Total occupants
    if all(col in df.columns for col in ['num_adults', 'num_children']):
        try:
            # Convert to float and handle missing values
            adults = pd.to_numeric(df['num_adults'], errors='coerce').fillna(0)
            children = pd.to_numeric(df['num_children'], errors='coerce').fillna(0)
            df['total_occupants'] = adults + children
        except Exception as e:
            # Log the error but continue processing
            logger.error(f"Error calculating total occupants: {str(e)}")
    
    logger.info("Extracted additional features")
    
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and converting data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    try:
        # Handle no_show column to ensure it's in the correct format
        if 'no_show' in df.columns:
            # Convert to numeric, handle any non-numeric values
            df['no_show'] = pd.to_numeric(df['no_show'], errors='coerce')
            # Fill any missing values with 0 (assuming no-show is less common) - avoid inplace
            df['no_show'] = df['no_show'].fillna(0)
            # Ensure it's always 0 or 1 (binary)
            df['no_show'] = df['no_show'].apply(lambda x: 1 if x > 0 else 0)
        
        # Convert data types for numeric columns
        numeric_cols = ['arrival_day', 'checkout_day', 'num_adults', 'num_children']
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Handle missing values appropriately for each column - avoid inplace
                if col == 'arrival_day' or col == 'checkout_day':
                    # For day columns, use the median if available, otherwise use 15 (middle of month)
                    median_val = df[col].median()
                    df[col] = df[col].fillna(15 if pd.isna(median_val) else median_val)
                elif col == 'num_children':
                    # For children, assume 0 if missing
                    df[col] = df[col].fillna(0)
                elif col == 'num_adults':
                    # For adults, use median or default to 1
                    median_val = df[col].median()
                    df[col] = df[col].fillna(1 if pd.isna(median_val) else median_val)
        
        # Replace any infinities with NaN, then fill with appropriate values
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    # If median is also NaN, use a reasonable default based on the column
                    if 'day' in col:
                        df[col] = df[col].fillna(15)  # Middle of month
                    elif 'num' in col or 'count' in col:
                        df[col] = df[col].fillna(0)   # Default count
                    else:
                        df[col] = df[col].fillna(0)   # Generic default
                else:
                    df[col] = df[col].fillna(median_val)
        
        # Handle categorical columns - avoid inplace
        for col in df.select_dtypes(include=['object']).columns:
            # Fill missing values with 'Unknown'
            df[col] = df[col].fillna('Unknown')
        
        # Check for and remove duplicate rows
        if df.duplicated().any():
            initial_count = len(df)
            df = df.drop_duplicates()
            logger.info(f"Removed {initial_count - len(df)} duplicate rows")
        
        # Check for and drop rows with too many missing values
        # Using a threshold of 50% missing values
        threshold = len(df.columns) * 0.5
        rows_before = len(df)
        df = df.dropna(thresh=threshold)
        if rows_before > len(df):
            logger.info(f"Dropped {rows_before - len(df)} rows with more than 50% missing values")
            
        logger.info("Cleaned dataset")
        
    except Exception as e:
        logger.error(f"Error in clean_dataset: {str(e)}")
        # If an error occurs, return the original DataFrame to prevent pipeline failure
    
    return df 