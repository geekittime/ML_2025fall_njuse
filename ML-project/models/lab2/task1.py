"""
Task 1: Neural Network Models for PR Time-to-Close Prediction (Regression)

Supports multiple architectures: MLP, Wide&Deep, DeepCross, Shared-Bottom, MMoE
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import (
    get_data_path, LAB2_CONFIG, RANDOM_SEED, 
    TRAIN_SPLIT_RATIO, get_checkpoint_path
)
from utils.data_loader import load_and_merge_data, prepare_features, train_test_split_by_time
from models.lab2.base_model import Lab2BaseModel
from models.lab2.architectures import (
    MLPRegressor, WideAndDeepRegressor, DeepCrossRegressor,
    SharedBottomRegressor, MMoERegressor
)


class Task1RegressionModel(Lab2BaseModel):
    """Unified model class for Task 1 Regression"""
    
    def __init__(self, model_type="mlp", config=None, random_seed=RANDOM_SEED):
        """
        Args:
            model_type: Type of model ("mlp", "wide_deep", "deep_cross", "shared_bottom", "mmoe")
            config: Model-specific configuration
            random_seed: Random seed
        """
        training_config = LAB2_CONFIG.get("training", {})
        if config:
            training_config.update(config)
        
        super().__init__(
            model_name=f"task1_{model_type}",
            task="regression",
            config=training_config,
            random_seed=random_seed
        )
        
        self.model_type = model_type
        self.model_config = LAB2_CONFIG["task1"].get(model_type, {})
    
    def build_model(self, input_dim: int):
        """Build the specified model architecture"""
        if self.model_type == "mlp":
            self.model = MLPRegressor(
                input_dim=input_dim,
                hidden_layers=self.model_config.get("hidden_layers", [128, 64, 32]),
                dropout_rate=self.model_config.get("dropout_rate", 0.2)
            ).to(self.device)
        
        elif self.model_type == "wide_deep":
            self.model = WideAndDeepRegressor(
                input_dim=input_dim,
                deep_dims=self.model_config.get("deep_dims", [128, 64]),
                dropout_rate=self.model_config.get("dropout_rate", 0.2)
            ).to(self.device)
        
        elif self.model_type == "deep_cross":
            self.model = DeepCrossRegressor(
                input_dim=input_dim,
                num_cross_layers=self.model_config.get("num_cross_layers", 3),
                deep_dims=self.model_config.get("deep_dims", [128, 64]),
                dropout_rate=self.model_config.get("dropout_rate", 0.2)
            ).to(self.device)
        
        elif self.model_type == "shared_bottom":
            self.model = SharedBottomRegressor(
                input_dim=input_dim,
                shared_dims=self.model_config.get("shared_dims", [128, 64]),
                task_dims=self.model_config.get("task_dims", [32]),
                num_tasks=self.model_config.get("num_tasks", 1),
                dropout_rate=self.model_config.get("dropout_rate", 0.2)
            ).to(self.device)
        
        elif self.model_type == "mmoe":
            self.model = MMoERegressor(
                input_dim=input_dim,
                num_experts=self.model_config.get("num_experts", 3),
                expert_dims=self.model_config.get("expert_dims", [64, 32]),
                task_dims=self.model_config.get("task_dims", [32]),
                num_tasks=self.model_config.get("num_tasks", 1),
                dropout_rate=self.model_config.get("dropout_rate", 0.2)
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"\n{self.model_type.upper()} Architecture:")
        print(self.model)
        
        return self.model


def main(model_type="mlp", dataset_name="yii2", use_extracted=True):
    """
    Main function to train and evaluate Task 1 regression models
    
    Args:
        model_type: Type of model to train
        dataset_name: Dataset to use
        use_extracted: Whether to use pre-extracted features
    """
    print("="*70)
    print(f"Task 1: {model_type.upper()} for PR Time-to-Close Prediction")
    print("="*70)
    
    # 1. Load data
    data_path = get_data_path(dataset_name)
    df = load_and_merge_data(data_path, use_extracted=use_extracted)
    
    # 2. Prepare features for regression
    X, y, features, df_filtered = prepare_features(
        df,
        task="regression",
        apply_log_transform=True,
        max_ttc_hours=1000
    )
    
    # 3. Split data by time
    X_train, X_test, y_train, y_test = train_test_split_by_time(
        X, y, df_filtered, split_ratio=TRAIN_SPLIT_RATIO
    )
    
    # 4. Initialize model
    model = Task1RegressionModel(model_type=model_type)
    
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
    print(f"\nTraining {model_type.upper()} model...")
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
    save_path = get_checkpoint_path(f"task1_{model_type}", dataset_name)
    model.save_model(save_path)
    
    print(f"\n{'='*70}")
    print("Task 1 Training Completed!")
    print(f"{'='*70}")
    print(f"Model: {model_type.upper()}")
    print(f"Dataset: {dataset_name}")
    print(f"MAE: {metrics['mae']:.2f} hours ({metrics['mae']/24:.2f} days)")
    print(f"RMSE: {metrics['rmse']:.2f} hours ({metrics['rmse']/24:.2f} days)")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*70}\n")
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Task 1 Regression Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  mlp          - Multi-Layer Perceptron
  wide_deep    - Wide & Deep Network
  deep_cross   - Deep & Cross Network
  shared_bottom - Shared-Bottom Multi-Task (single task mode)
  mmoe         - Multi-gate Mixture-of-Experts (single task mode)

Examples:
  python task1.py --model mlp --dataset yii2
  python task1.py --model wide_deep --dataset django
  python task1.py --model deep_cross --dataset tensorflow --no-extracted
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "wide_deep", "deep_cross", "shared_bottom", "mmoe"],
        help="Model architecture to use"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="yii2",
        help="Dataset name (default: yii2)"
    )
    
    parser.add_argument(
        "--no-extracted",
        action="store_true",
        help="Don't use pre-extracted features"
    )
    
    args = parser.parse_args()
    
    model, metrics = main(
        model_type=args.model,
        dataset_name=args.dataset,
        use_extracted=not args.no_extracted
    )
