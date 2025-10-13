"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


def load_and_merge_data(
    data_path: Path,
    use_extracted: bool = False
) -> pd.DataFrame:
    """
    Load and merge PR data from multiple Excel files
    
    Args:
        data_path: Path to the data directory
        use_extracted: If True, load pre-extracted features file
    
    Returns:
        Merged DataFrame with all PR information
    """
    print(f"Loading data from: {data_path}")
    
    if use_extracted:
        extracted_path = data_path / 'PR_extracted_features.xlsx'
        if extracted_path.exists():
            print("Loading pre-extracted features...")
            merged_df = pd.read_excel(extracted_path)
            print(f"Loaded {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
            return merged_df
        else:
            print("Pre-extracted file not found, loading from source files...")
    
    # Load individual files
    pr_info = pd.read_excel(data_path / 'PR_info.xlsx')
    pr_features = pd.read_excel(data_path / 'PR_features.xlsx')
    author_features = pd.read_excel(data_path / 'author_features.xlsx')
    pr_info_conversation = pd.read_excel(data_path / 'PR_info_add_conversation.xlsx')
    
    # Merge dataframes
    merged_df = pd.merge(pr_info, pr_features, on='number', how='left')
    merged_df = pd.merge(merged_df, author_features, on='number', how='left')
    merged_df = pd.merge(merged_df, pr_info_conversation, on='number', how='left')
    
    # Handle duplicate columns
    merged_df = _handle_duplicate_columns(merged_df)
    
    # Clean up unnamed index columns
    columns_to_drop = [col for col in merged_df.columns if 'Unnamed: 0' in str(col)]
    merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    print(f"Data merged: {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
    
    return merged_df


def _handle_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicate columns from merging (e.g., created_at_x, created_at_y)"""
    for col in df.columns:
        if col.endswith('_y'):
            df.drop(columns=[col], inplace=True)
        elif col.endswith('_x'):
            df.rename(columns={col: col[:-2]}, inplace=True)
    return df


def prepare_features(
    df: pd.DataFrame,
    task: str = "regression",
    target_col: str = None,
    features_to_remove: List[str] = None,
    apply_log_transform: bool = True,
    max_ttc_hours: Optional[float] = 1000
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features for machine learning models
    
    Args:
        df: Input DataFrame
        task: "regression" or "classification"
        target_col: Name of target column (if None, auto-determine based on task)
        features_to_remove: List of column names to exclude from features
        apply_log_transform: Whether to apply log1p transform to target (regression only)
        max_ttc_hours: Maximum TTC hours to filter outliers (None to disable)
    
    Returns:
        Tuple of (X features DataFrame, y target Series, feature names list)
    """
    df = df.copy()
    
    # Convert datetime columns
    df['created_at'] = pd.to_datetime(df['created_at'])
    if 'closed_at' in df.columns:
        df['closed_at'] = pd.to_datetime(df['closed_at'])
    
    # Prepare target variable based on task
    if task == "regression":
        # Calculate TTC (Time To Close) in hours
        df.dropna(subset=['closed_at', 'created_at'], inplace=True)
        df['TTC_hours'] = (df['closed_at'] - df['created_at']).dt.total_seconds() / 3600
        df = df[df['TTC_hours'] >= 0]
        
        # Filter outliers if specified
        if max_ttc_hours is not None:
            df = df[df['TTC_hours'] <= max_ttc_hours]
        
        # Apply log transform
        if apply_log_transform:
            df['log_TTC_hours'] = np.log1p(df['TTC_hours'])
            target_col = 'log_TTC_hours'
            print(f"Applied log transform to TTC. Using '{target_col}' as target.")
        else:
            target_col = 'TTC_hours'
        
        y = df[target_col]
        
    elif task == "classification":
        # For classification, typically predicting 'merged' status
        if target_col is None:
            target_col = 'merged'
        
        # Filter to closed PRs
        if 'state' in df.columns:
            df = df[df['state'] == 'closed']
            print(f"Filtered to closed PRs: {len(df)} rows")
        
        y = df[target_col].astype(int)
    
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")
    
    # Select numeric features
    features = df.select_dtypes(include=np.number).columns.tolist()
    
    # Remove specified features
    default_remove = ['number', 'author_id', 'project_id', 'TTC_hours', 'log_TTC_hours']
    if features_to_remove:
        default_remove.extend(features_to_remove)
    
    # Also remove target column if it's in features
    if target_col in features:
        default_remove.append(target_col)
    
    features = [f for f in features if f not in default_remove]
    
    X = df[features]
    
    # Handle infinite values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill missing values with median
    X.fillna(X.median(), inplace=True)
    
    print(f"Prepared {len(features)} features for {task} task")
    print(f"Target variable: {target_col}, shape: {y.shape}")
    
    return X, y, features


def train_test_split_by_time(
    X: pd.DataFrame,
    y: pd.Series,
    df_full: pd.DataFrame,
    split_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test by temporal order
    
    Args:
        X: Feature DataFrame
        y: Target Series
        df_full: Full DataFrame containing 'created_at' column
        split_ratio: Ratio of training data (0-1)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Sort by creation time
    df_full = df_full.sort_values('created_at')
    
    # Align X and y with sorted indices
    X = X.loc[df_full.index]
    y = y.loc[df_full.index]
    
    # Split
    split_point = int(len(X) * split_ratio)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"\nTrain/Test split by time:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test data loading
    from config import get_data_path
    
    data_path = get_data_path("yii2")
    df = load_and_merge_data(data_path)
    
    print("\n" + "="*50)
    print("Testing regression task preparation:")
    X, y, features = prepare_features(df, task="regression")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {features[:5]}...")
    
    print("\n" + "="*50)
    print("Testing classification task preparation:")
    X, y, features = prepare_features(df, task="classification")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
