"""
Task 1: Linear Regression for PR Time-to-Close Prediction
Refactored version using Lab1BaseModel
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import Ridge
from config import get_data_path, LAB1_CONFIG, RANDOM_SEED, TRAIN_SPLIT_RATIO
from utils.data_loader import load_and_merge_data, prepare_features, train_test_split_by_time
from models.lab1.base_model import Lab1BaseModel


class LinearRegressionModel(Lab1BaseModel):
    """Linear Regression (Ridge) model for Task 1"""
    
    def __init__(self, config=None, random_seed=RANDOM_SEED):
        super().__init__(
            model_type="linear_regression",
            task="regression",
            config=config or LAB1_CONFIG["linear_regression"],
            random_seed=random_seed
        )
    
    def build_model(self):
        """Build Ridge regression model"""
        self.model = Ridge(
            alpha=self.config.get("alpha", 1.0),
            random_state=self.random_seed
        )
        return self.model


def main(dataset_name="yii2"):
    """
    Main function to train and evaluate linear regression model
    
    Args:
        dataset_name: Name of dataset to use
    """
    print("="*60)
    print("Task 1: Linear Regression for PR Time-to-Close Prediction")
    print("="*60)
    
    # 1. Load data
    data_path = get_data_path(dataset_name)
    df = load_and_merge_data(data_path, use_extracted=False)
    
    # 2. Prepare features
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
    model = LinearRegressionModel()
    model.build_model()
    
    # 5. Preprocess data
    X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test, scale=True)
    
    # 6. Train model
    model.train(X_train_scaled, y_train)
    
    # 7. Evaluate model
    metrics = model.evaluate(X_test_scaled, y_test, verbose=True)
    
    # 8. Analyze feature importance
    model.analyze_feature_importance(top_n=10)
    
    # 9. Save model (optional)
    # from config import get_checkpoint_path
    # save_path = get_checkpoint_path("task1_linear", dataset_name)
    # model.save_model(save_path)
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Linear Regression for Task 1")
    parser.add_argument("--dataset", type=str, default="yii2", 
                        help="Dataset name (default: yii2)")
    args = parser.parse_args()
    
    model, metrics = main(args.dataset)
