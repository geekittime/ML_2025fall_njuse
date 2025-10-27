"""
Task 2: Neural Network Models for PR Merge Status Prediction (Classification)
Supports multiple architectures: MLP, Wide&Deep, DeepCross
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from config import (
    get_data_path, LAB2_CONFIG, RANDOM_SEED, 
    TRAIN_SPLIT_RATIO, get_checkpoint_path
)
from utils.data_loader import load_and_merge_data, prepare_features, train_test_split_by_time
from models.lab2.base_model import Lab2BaseModel
from models.lab2.architectures import (
    MLPClassifier, WideAndDeepClassifier, DeepCrossClassifier
)


class Task2ClassificationModel(Lab2BaseModel):
    """Unified model class for Task 2 Classification"""
    
    def __init__(self, model_type="mlp", config=None, random_seed=RANDOM_SEED):
        """
        Args:
            model_type: Type of model ("mlp", "wide_deep", "deep_cross")
            config: Model-specific configuration
            random_seed: Random seed
        """
        training_config = LAB2_CONFIG.get("training", {})
        if config:
            training_config.update(config)
        
        super().__init__(
            model_name=f"task2_{model_type}",
            task="classification",
            config=training_config,
            random_seed=random_seed
        )
        
        self.model_type = model_type
        self.model_config = LAB2_CONFIG["task2"].get(model_type, {})
    
    def build_model(self, input_dim: int):
        """Build the specified model architecture"""
        if self.model_type == "mlp":
            self.model = MLPClassifier(
                input_dim=input_dim,
                hidden_layers=self.model_config.get("hidden_layers", [128, 64, 32]),
                dropout_rate=self.model_config.get("dropout_rate", 0.2)
            ).to(self.device)
        
        elif self.model_type == "wide_deep":
            self.model = WideAndDeepClassifier(
                input_dim=input_dim,
                deep_dims=self.model_config.get("deep_dims", [128, 64]),
                dropout_rate=self.model_config.get("dropout_rate", 0.2)
            ).to(self.device)
        
        elif self.model_type == "deep_cross":
            self.model = DeepCrossClassifier(
                input_dim=input_dim,
                num_cross_layers=self.model_config.get("num_cross_layers", 3),
                deep_dims=self.model_config.get("deep_dims", [128, 64]),
                dropout_rate=self.model_config.get("dropout_rate", 0.2)
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"\n{self.model_type.upper()} Architecture:")
        print(self.model)
        
        return self.model
    
    def _evaluate_classification(self, y_true, y_pred_prob, verbose=True):
        """Override to add more detailed classification metrics"""
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro,
            'confusion_matrix': cm
        }
        
        if verbose:
            print("\n" + "="*70)
            print("Task 2: Classification Model Evaluation Results")
            print("="*70)
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precision (Macro): {precision:.4f}")
            print(f"Recall (Macro): {recall:.4f}")
            print(f"F1 Score (Macro): {f1_macro:.4f}")
            print("\nConfusion Matrix:")
            print(cm)
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred, target_names=['Not Merged', 'Merged'], zero_division=0))
            print("="*70)
        
        return metrics


def main(model_type="mlp", dataset_name="yii2", use_extracted=True):
    """
    Main function to train and evaluate Task 2 classification models
    
    Args:
        model_type: Type of model to train
        dataset_name: Dataset to use
        use_extracted: Whether to use pre-extracted features
    """
    print("="*70)
    print(f"Task 2: {model_type.upper()} for PR Merge Status Prediction")
    print("="*70)
    
    # 1. Load data
    data_path = get_data_path(dataset_name)
    df = load_and_merge_data(data_path, use_extracted=use_extracted)
    
    # 2. Prepare features for classification
    X, y, features, df_filtered = prepare_features(
        df,
        task="classification",
        target_col="merged"
    )
    
    # 3. Split data by time
    X_train, X_test, y_train, y_test = train_test_split_by_time(
        X, y, df_filtered, split_ratio=TRAIN_SPLIT_RATIO
    )
    
    # 4. Initialize model
    model = Task2ClassificationModel(model_type=model_type)
    
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
    save_path = get_checkpoint_path(f"task2_{model_type}", dataset_name)
    model.save_model(save_path)
    
    print(f"\n{'='*70}")
    print("Task 2 Training Completed!")
    print(f"{'='*70}")
    print(f"Model: {model_type.upper()}")
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision (Macro): {metrics['precision']:.4f}")
    print(f"Recall (Macro): {metrics['recall']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*70}\n")
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Task 2 Classification Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  mlp          - Multi-Layer Perceptron
  wide_deep    - Wide & Deep Network
  deep_cross   - Deep & Cross Network

Examples:
  python task2.py --model mlp --dataset yii2
  python task2.py --model wide_deep --dataset django
  python task2.py --model deep_cross --dataset tensorflow --no-extracted
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "wide_deep", "deep_cross"],
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
