"""
Task 1: MLP (Multi-Layer Perceptron) for PR Time-to-Close Prediction
Refactored version using Lab2BaseModel
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from config import (
    get_data_path, LAB2_CONFIG, RANDOM_SEED, 
    TRAIN_SPLIT_RATIO, get_checkpoint_path
)
from utils.data_loader import load_and_merge_data, prepare_features, train_test_split_by_time
from models.lab2.base_model import Lab2BaseModel


class MLPRegressor(nn.Module):
    """Multi-Layer Perceptron for Regression"""
    
    def __init__(self, input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.2):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class LinearMLPModel(Lab2BaseModel):
    """MLP model for Task 1 Regression"""
    
    def __init__(self, config=None, random_seed=RANDOM_SEED):
        training_config = LAB2_CONFIG.get("training", {})
        if config:
            training_config.update(config)
        
        super().__init__(
            model_name="linear_mlp",
            task="regression",
            config=training_config,
            random_seed=random_seed
        )
        
        self.mlp_config = LAB2_CONFIG.get("linear_mlp", {})
    
    def build_model(self, input_dim: int):
        """Build MLP model"""
        hidden_layers = self.mlp_config.get("hidden_layers", [128, 64, 32])
        dropout_rate = self.mlp_config.get("dropout_rate", 0.2)
        
        self.model = MLPRegressor(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        print(f"\nMLP Architecture:")
        print(self.model)
        
        return self.model


def main(dataset_name="yii2", use_extracted=True):
    """
    Main function to train and evaluate MLP model
    
    Args:
        dataset_name: Name of dataset to use
        use_extracted: Whether to use pre-extracted features
    """
    print("="*60)
    print("Task 1: MLP for PR Time-to-Close Prediction")
    print("="*60)
    
    # 1. Load data
    data_path = get_data_path(dataset_name)
    df = load_and_merge_data(data_path, use_extracted=use_extracted)
    
    # 2. Prepare features
    # For Lab2, we often use extracted features with additional preprocessing
    X, y, features = prepare_features(
        df,
        task="regression",
        apply_log_transform=True,
        max_ttc_hours=1000
    )
    
    # 3. Split data by time
    X_train, X_test, y_train, y_test = train_test_split_by_time(
        X, y, df, split_ratio=TRAIN_SPLIT_RATIO
    )
    
    # 4. Initialize model
    model = LinearMLPModel()
    
    # 5. Preprocess data
    X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test)
    
    # 6. Build model
    input_dim = X_train_scaled.shape[1]
    model.build_model(input_dim)
    
    # 7. Prepare DataLoaders
    batch_size = model.config.get("batch_size", 64)
    train_loader, test_loader = model.prepare_dataloaders(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        batch_size=batch_size
    )
    
    # 8. Train model
    training_history = model.train(
        train_loader=train_loader,
        val_loader=test_loader,
        n_epochs=model.config.get("n_epochs", 100),
        learning_rate=model.config.get("learning_rate", 0.001),
        patience=model.config.get("patience", 10),
        verbose=True
    )
    
    # 9. Evaluate model
    metrics = model.evaluate(test_loader, y_test, verbose=True)
    
    # 10. Save model
    save_path = get_checkpoint_path("task1_linear_mlp", dataset_name)
    model.save_model(save_path)
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MLP for Task 1")
    parser.add_argument("--dataset", type=str, default="yii2", 
                        help="Dataset name (default: yii2)")
    parser.add_argument("--no-extracted", action="store_true",
                        help="Don't use pre-extracted features")
    args = parser.parse_args()
    
    model, metrics = main(args.dataset, use_extracted=not args.no_extracted)
