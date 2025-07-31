"""
Utility functions for the Yango Accra Mobility Prediction project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List, Dict
import config

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data files with error handling
    
    Returns:
        Tuple of (train_df, test_df, weather_df, sample_submission)
    """
    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        test_df = pd.read_csv(config.TEST_FILE)
        weather_df = pd.read_csv(config.WEATHER_FILE)
        sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
        
        print("Data loaded successfully!")
        print(f"Train: {train_df.shape}, Test: {test_df.shape}")
        print(f"Weather: {weather_df.shape}, Sample: {sample_submission.shape}")
        
        return train_df, test_df, weather_df, sample_submission
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure all data files are in the data/ directory")
        raise

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Calculate and display model evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for display
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    print(f"\n{model_name} Performance:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predictions vs Actual"):
    """
    Plot predicted vs actual values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_names: List[str], importance_values: np.ndarray, 
                          title: str = "Feature Importance"):
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_values: Importance scores
        title: Plot title
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return importance_df

def create_submission_file(test_ids: pd.Series, predictions: np.ndarray, 
                         filename: str = "submission.csv") -> pd.DataFrame:
    """
    Create submission file
    
    Args:
        test_ids: Test trip IDs
        predictions: Model predictions
        filename: Output filename
    
    Returns:
        Submission dataframe
    """
    submission = pd.DataFrame({
        'trip_id': test_ids,
        'Target': predictions
    })
    
    output_path = config.get_output_path(filename)
    submission.to_csv(output_path, index=False)
    
    print(f"Submission file saved: {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
    
    return submission

def data_summary(df: pd.DataFrame, name: str = "Dataset"):
    """
    Print comprehensive data summary
    
    Args:
        df: DataFrame to summarize
        name: Dataset name for display
    """
    print(f"\n{name} Summary:")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values")
    
    print(f"\nNumerical features summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())

def quick_eda(df: pd.DataFrame, target_col: str = None):
    """
    Quick exploratory data analysis
    
    Args:
        df: DataFrame to analyze
        target_col: Target column name for correlation analysis
    """
    data_summary(df)
    
    if target_col and target_col in df.columns:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        df[target_col].hist(bins=50)
        plt.title(f'{target_col} Distribution')
        plt.xlabel(target_col)
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        df.boxplot(column=target_col)
        plt.title(f'{target_col} Boxplot')
        
        plt.subplot(1, 3, 3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        correlations[1:11].plot(kind='barh')
        plt.title('Top 10 Correlations with Target')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{target_col} Statistics:")
        print(f"Mean: {df[target_col].mean():.2f}")
        print(f"Median: {df[target_col].median():.2f}")
        print(f"Std: {df[target_col].std():.2f}")
        print(f"Skewness: {df[target_col].skew():.2f}")

if __name__ == "__main__":
    print("Utility functions for Yango Accra Mobility Prediction")
    print("Import this module to use the helper functions")
