"""
Configuration file for ML-project
Centralized configuration for data paths, model hyperparameters, and training settings
"""
import os
from pathlib import Path

# ==================== Project Paths ====================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
UTILS_DIR = PROJECT_ROOT / "utils"

# Create directories if they don't exist
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# ==================== Data Configuration ====================
# Available datasets
DATASETS = [
    "yii2", "django", "moby", "opencv", "react", 
    "salt", "scikit-learn", "symfony", "tensorflow", "terraform"
]

# Default dataset
DEFAULT_DATASET = "yii2"

# Data files configuration
DATA_FILES = {
    "pr_info": "PR_info.xlsx",
    "pr_features": "PR_features.xlsx",
    "author_features": "author_features.xlsx",
    "pr_info_conversation": "PR_info_add_conversation.xlsx",
    "pr_comment_info": "PR_comment_info.xlsx",
    "pr_extracted": "PR_extracted_features.xlsx"
}

# ==================== Model Configuration ====================
# Random seed for reproducibility
RANDOM_SEED = 42

# Train-test split ratio
TRAIN_SPLIT_RATIO = 0.8

# ==================== Lab1 Configuration (Classical ML) ====================
LAB1_CONFIG = {
    # Linear Regression
    "linear_regression": {
        "alpha": 1.0,  # Ridge regression parameter
        "max_iter": 1000,
    },
    
    # Polynomial Regression
    "polynomial_regression": {
        "degree": 2,
        "alpha": 1.0,
    },
    
    # Random Forest Regression
    "random_forest_regression": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    
    # Logistic Regression
    "logistic_regression": {
        "class_weight": "balanced",
        "max_iter": 3000,
        "random_state": RANDOM_SEED,
    },
    
    # Random Forest Classification
    "random_forest_classification": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
}

# ==================== Lab2 Configuration (Deep Learning) ====================
LAB2_CONFIG = {
    # General training parameters
    "training": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "n_epochs": 100,
        "patience": 10,  # Early stopping patience
        "device": "auto",  # "auto", "cuda", or "cpu"
    },
    
    # Task 1: Regression models
    "task1": {
        # MLP Regressor
        "mlp": {
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.2,
        },
        
        # Wide & Deep Regressor
        "wide_deep": {
            "deep_dims": [128, 64],
            "dropout_rate": 0.2,
        },
        
        # Deep & Cross Regressor
        "deep_cross": {
            "num_cross_layers": 3,
            "deep_dims": [128, 64],
            "dropout_rate": 0.2,
        },
        
        # Shared-Bottom (can be used for single task)
        "shared_bottom": {
            "shared_dims": [128, 64],
            "task_dims": [32],
            "num_tasks": 1,
            "dropout_rate": 0.2,
        },
        
        # MMoE (can be used for single task)
        "mmoe": {
            "num_experts": 3,
            "expert_dims": [64, 32],
            "task_dims": [32],
            "num_tasks": 1,
            "dropout_rate": 0.2,
        },
    },
    
    # Task 2: Classification models
    "task2": {
        # MLP Classifier
        "mlp": {
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.2,
        },
        
        # Wide & Deep Classifier
        "wide_deep": {
            "deep_dims": [128, 64],
            "dropout_rate": 0.2,
        },
        
        # Deep & Cross Classifier
        "deep_cross": {
            "num_cross_layers": 3,
            "deep_dims": [128, 64],
            "dropout_rate": 0.2,
        },
    },
    
    # Multi-Task Learning
    "multitask": {
        "shared_dims": [128, 64],
        "regression_dims": [32],
        "classification_dims": [32],
        "dropout_rate": 0.2,
        "loss_weights": {
            "regression": 1.0,
            "classification": 1.0,
        }
    }
}

# ==================== Feature Engineering ====================
FEATURE_CONFIG = {
    # Features to remove (IDs, target variables)
    "features_to_remove": ["number", "author_id", "project_id"],
    
    # Target variable transformations
    "apply_log_transform": True,
    
    # Feature scaling
    "scaling_method": "standard",  # "standard", "minmax", or "none"
    
    # Missing value handling
    "missing_value_strategy": "median",  # "median", "mean", or "drop"
    
    # Outlier handling for TTC (in hours)
    "max_ttc_hours": 1000,
}

# ==================== Logging Configuration ====================
LOGGING_CONFIG = {
    "verbose": True,
    "log_file": PROJECT_ROOT / "training.log",
    "tensorboard_dir": PROJECT_ROOT / "runs",
}

# ==================== Evaluation Metrics ====================
# Task 1: Regression metrics
REGRESSION_METRICS = ["mae", "rmse", "r2"]

# Task 2: Classification metrics
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1"]


def get_data_path(dataset_name: str, file_key: str = None) -> Path:
    """
    Get the path to a data file for a specific dataset
    
    Args:
        dataset_name: Name of the dataset (e.g., "yii2")
        file_key: Key from DATA_FILES dict (e.g., "pr_info"), if None returns dataset dir
    
    Returns:
        Path object to the data file or directory
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {DATASETS}")
    
    dataset_path = DATA_DIR / dataset_name
    
    if file_key is None:
        return dataset_path
    
    if file_key not in DATA_FILES:
        raise ValueError(f"Unknown file key: {file_key}. Available: {list(DATA_FILES.keys())}")
    
    return dataset_path / DATA_FILES[file_key]


def get_checkpoint_path(model_name: str, dataset_name: str = None) -> Path:
    """
    Get the path to save/load model checkpoint
    
    Args:
        model_name: Name of the model
        dataset_name: Optional dataset name to include in filename
    
    Returns:
        Path object to the checkpoint file
    """
    if dataset_name:
        filename = f"{model_name}_{dataset_name}.pth"
    else:
        filename = f"{model_name}.pth"
    
    return CHECKPOINTS_DIR / filename


if __name__ == "__main__":
    # Test configuration
    print("Project Configuration Test")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Checkpoints Directory: {CHECKPOINTS_DIR}")
    print(f"\nAvailable Datasets: {', '.join(DATASETS)}")
    print(f"\nExample data path: {get_data_path('yii2', 'pr_info')}")
    print(f"Example checkpoint path: {get_checkpoint_path('task1_linear', 'yii2')}")
