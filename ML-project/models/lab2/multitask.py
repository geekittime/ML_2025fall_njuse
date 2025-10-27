"""
Multi-Task Learning: Joint training for both Task 1 (Regression) and Task 2 (Classification)
Predicts both PR Time-to-Close and Merge Status simultaneously
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from config import (
    get_data_path, LAB2_CONFIG, RANDOM_SEED, 
    TRAIN_SPLIT_RATIO, get_checkpoint_path
)
from utils.data_loader import load_and_merge_data, train_test_split_by_time
from models.lab2.architectures import MultiTaskModel


class MultiTaskLearning:
    """Multi-Task Learning model wrapper"""
    
    def __init__(self, config=None, random_seed=RANDOM_SEED):
        self.config = config or LAB2_CONFIG["multitask"]
        self.training_config = LAB2_CONFIG["training"]
        self.random_seed = random_seed
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Device configuration
        device_config = self.training_config.get("device", "auto")
        if device_config == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_config)
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.training_history = {
            'train_loss_reg': [], 'train_loss_cls': [], 'train_loss_total': [],
            'val_loss_reg': [], 'val_loss_cls': [], 'val_loss_total': []
        }
    
    def prepare_multitask_data(self, df):
        """
        Prepare data for multi-task learning
        Returns both regression and classification targets
        """
        import warnings
        warnings.filterwarnings('ignore')
        
        print("\nPreparing data for multi-task learning...")
        
        # Convert boolean columns to int
        for bool_col in df.select_dtypes(include=bool).columns.tolist():
            df[bool_col] = df[bool_col].astype(int)
        
        # Process datetime columns
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['closed_at'] = pd.to_datetime(df['closed_at'])
        
        # Calculate TTC for regression task
        df['TTC_hours'] = (df['closed_at'] - df['created_at']).dt.total_seconds() / 3600
        df.dropna(subset=['closed_at', 'created_at'], inplace=True)
        df = df[(df['TTC_hours'] >= 0) & (df['TTC_hours'] <= 1000)]
        df['log_TTC_hours'] = np.log1p(df['TTC_hours'])
        
        # Apply log transform to time-based features if they exist
        if 'last_pr_update' in df.columns:
            df['log_last_pr_update'] = np.log1p(df['last_pr_update'])
        if 'last_comment_update' in df.columns:
            df['log_last_comment_update'] = np.log1p(df['last_comment_update'])
        
        # Extract targets
        y_regression = df['log_TTC_hours']
        y_classification = df['merged'].astype(int)
        
        # Select features
        features = df.select_dtypes(include=np.number).columns.tolist()
        features_to_remove = [
            'number', 'merged', 'TTC_hours', 'log_TTC_hours',
            'last_pr_update', 'last_comment_update', 
            'log_last_pr_update', 'log_last_comment_update'
        ]
        features = [f for f in features if f not in features_to_remove]
        
        X = df[features]
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
        
        # Clean targets
        y_regression.replace([np.inf, -np.inf], np.nan, inplace=True)
        y_regression.fillna(y_regression.median(), inplace=True)
        
        print(f"Features: {len(features)}")
        print(f"Samples: {len(X)}")
        print(f"Regression target (log_TTC_hours) range: [{y_regression.min():.2f}, {y_regression.max():.2f}]")
        print(f"Classification target (merged) distribution:\n{y_classification.value_counts()}")
        
        return X, y_regression, y_classification, features, df
    
    def build_model(self, input_dim):
        """Build multi-task model"""
        self.model = MultiTaskModel(
            input_dim=input_dim,
            shared_dims=self.config.get("shared_dims", [128, 64]),
            regression_dims=self.config.get("regression_dims", [32]),
            classification_dims=self.config.get("classification_dims", [32]),
            dropout_rate=self.config.get("dropout_rate", 0.2)
        ).to(self.device)
        
        print("\nMulti-Task Model Architecture:")
        print(self.model)
        
        return self.model
    
    def train(self, train_loader, val_loader, n_epochs=100, learning_rate=0.001, patience=10):
        """Train multi-task model"""
        loss_weights = self.config.get("loss_weights", {"regression": 1.0, "classification": 1.0})
        
        criterion_reg = nn.MSELoss()
        criterion_cls = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        print(f"\nTraining Multi-Task Model...")
        print(f"Loss weights - Regression: {loss_weights['regression']}, Classification: {loss_weights['classification']}")
        
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_loss_reg_sum, train_loss_cls_sum = 0.0, 0.0
            
            for inputs, labels_reg, labels_cls in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs_reg, outputs_cls = self.model(inputs)
                
                # Calculate losses
                loss_reg = criterion_reg(outputs_reg, labels_reg)
                loss_cls = criterion_cls(outputs_cls, labels_cls)
                
                # Combined loss with weights
                loss = loss_weights['regression'] * loss_reg + loss_weights['classification'] * loss_cls
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss_reg_sum += loss_reg.item() * inputs.size(0)
                train_loss_cls_sum += loss_cls.item() * inputs.size(0)
            
            avg_train_loss_reg = train_loss_reg_sum / len(train_loader.dataset)
            avg_train_loss_cls = train_loss_cls_sum / len(train_loader.dataset)
            avg_train_loss_total = avg_train_loss_reg + avg_train_loss_cls
            
            # Validation phase
            self.model.eval()
            val_loss_reg_sum, val_loss_cls_sum = 0.0, 0.0
            
            with torch.no_grad():
                for inputs, labels_reg, labels_cls in val_loader:
                    outputs_reg, outputs_cls = self.model(inputs)
                    loss_reg = criterion_reg(outputs_reg, labels_reg)
                    loss_cls = criterion_cls(outputs_cls, labels_cls)
                    
                    val_loss_reg_sum += loss_reg.item() * inputs.size(0)
                    val_loss_cls_sum += loss_cls.item() * inputs.size(0)
            
            avg_val_loss_reg = val_loss_reg_sum / len(val_loader.dataset)
            avg_val_loss_cls = val_loss_cls_sum / len(val_loader.dataset)
            avg_val_loss_total = avg_val_loss_reg + avg_val_loss_cls
            
            # Store history
            self.training_history['train_loss_reg'].append(avg_train_loss_reg)
            self.training_history['train_loss_cls'].append(avg_train_loss_cls)
            self.training_history['train_loss_total'].append(avg_train_loss_total)
            self.training_history['val_loss_reg'].append(avg_val_loss_reg)
            self.training_history['val_loss_cls'].append(avg_val_loss_cls)
            self.training_history['val_loss_total'].append(avg_val_loss_total)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Train Loss: Reg={avg_train_loss_reg:.4f} Cls={avg_train_loss_cls:.4f} Total={avg_train_loss_total:.4f} | "
                      f"Val Loss: Reg={avg_val_loss_reg:.4f} Cls={avg_val_loss_cls:.4f} Total={avg_val_loss_total:.4f}")
            
            # Early stopping
            if avg_val_loss_total < best_val_loss:
                best_val_loss = avg_val_loss_total
                epochs_no_improve = 0
                best_model_state = self.model.state_dict().copy()
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print("Loaded best model weights")
        
        return self.training_history
    
    def evaluate(self, test_loader, y_test_reg, y_test_cls):
        """Evaluate multi-task model"""
        self.model.eval()
        
        predictions_reg = []
        predictions_cls = []
        
        with torch.no_grad():
            for inputs, _, _ in test_loader:
                outputs_reg, outputs_cls = self.model(inputs)
                predictions_reg.append(outputs_reg.cpu().numpy())
                predictions_cls.append(torch.sigmoid(outputs_cls).cpu().numpy())
        
        predictions_reg = np.concatenate(predictions_reg).flatten()
        predictions_cls = np.concatenate(predictions_cls).flatten()
        
        # Regression metrics
        y_test_reg_original = np.expm1(y_test_reg)
        predictions_reg_original = np.expm1(predictions_reg)
        
        mae = mean_absolute_error(y_test_reg_original, predictions_reg_original)
        rmse = np.sqrt(mean_squared_error(y_test_reg_original, predictions_reg_original))
        r2 = r2_score(y_test_reg_original, predictions_reg_original)
        
        # Classification metrics
        y_pred_cls = (predictions_cls > 0.5).astype(int)
        accuracy = accuracy_score(y_test_cls, y_pred_cls)
        precision = precision_score(y_test_cls, y_pred_cls, average='macro', zero_division=0)
        recall = recall_score(y_test_cls, y_pred_cls, average='macro', zero_division=0)
        f1_macro = f1_score(y_test_cls, y_pred_cls, average='macro', zero_division=0)
        
        metrics = {
            'regression': {'mae': mae, 'rmse': rmse, 'r2': r2},
            'classification': {
                'accuracy': accuracy, 'precision': precision,
                'recall': recall, 'f1_macro': f1_macro
            }
        }
        
        print("\n" + "="*70)
        print("Multi-Task Learning Evaluation Results")
        print("="*70)
        print("\n【Task 1: Regression - TTC Prediction】")
        print(f"  MAE:  {mae:.2f} hours ({mae/24:.2f} days)")
        print(f"  RMSE: {rmse:.2f} hours ({rmse/24:.2f} days)")
        print(f"  R²:   {r2:.4f}")
        
        print("\n【Task 2: Classification - Merge Prediction】")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1_macro:.4f}")
        
        print("\n  Confusion Matrix:")
        print(confusion_matrix(y_test_cls, y_pred_cls))
        print("\n  Classification Report:")
        print(classification_report(y_test_cls, y_pred_cls, target_names=['Not Merged', 'Merged'], zero_division=0))
        print("="*70)
        
        return metrics


def main(dataset_name="yii2"):
    """Main function for multi-task learning"""
    print("="*70)
    print("Multi-Task Learning: Joint Training for Task 1 & Task 2")
    print("="*70)
    
    # 1. Load data
    data_path = get_data_path(dataset_name)
    df = load_and_merge_data(data_path, use_extracted=True)
    
    # 2. Initialize multi-task model
    mtl_model = MultiTaskLearning()
    
    # 3. Prepare multi-task data
    X, y_reg, y_cls, features, df_full = mtl_model.prepare_multitask_data(df)
    
    # 4. Split data by time
    from sklearn.preprocessing import StandardScaler
    
    df_full.sort_values('created_at', inplace=True)
    X = X.loc[df_full.index]
    y_reg = y_reg.loc[df_full.index]
    y_cls = y_cls.loc[df_full.index]
    
    split_point = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train_reg, y_test_reg = y_reg.iloc[:split_point], y_reg.iloc[split_point:]
    y_train_cls, y_test_cls = y_cls.iloc[:split_point], y_cls.iloc[split_point:]
    
    # 5. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain/Test split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # 6. Prepare PyTorch DataLoaders
    X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32)).to(mtl_model.device)
    y_train_reg_tensor = torch.tensor(y_train_reg.values.astype(np.float32)).to(mtl_model.device).view(-1, 1)
    y_train_cls_tensor = torch.tensor(y_train_cls.values.astype(np.float32)).to(mtl_model.device).view(-1, 1)
    
    X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32)).to(mtl_model.device)
    y_test_reg_tensor = torch.tensor(y_test_reg.values.astype(np.float32)).to(mtl_model.device).view(-1, 1)
    y_test_cls_tensor = torch.tensor(y_test_cls.values.astype(np.float32)).to(mtl_model.device).view(-1, 1)
    
    batch_size = mtl_model.training_config.get("batch_size", 64)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_reg_tensor, y_train_cls_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_reg_tensor, y_test_cls_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. Build model
    input_dim = X_train_scaled.shape[1]
    mtl_model.build_model(input_dim)
    
    # 8. Train model
    training_history = mtl_model.train(
        train_loader=train_loader,
        val_loader=test_loader,
        n_epochs=mtl_model.training_config.get("n_epochs", 100),
        learning_rate=mtl_model.training_config.get("learning_rate", 0.001),
        patience=mtl_model.training_config.get("patience", 10)
    )
    
    # 9. Evaluate model
    metrics = mtl_model.evaluate(test_loader, y_test_reg, y_test_cls)
    
    # 10. Save model
    save_path = get_checkpoint_path("multitask", dataset_name)
    torch.save({
        'model_state_dict': mtl_model.model.state_dict(),
        'config': mtl_model.config,
        'training_history': training_history,
        'features': features
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    
    return mtl_model, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Multi-Task Learning Model")
    parser.add_argument("--dataset", type=str, default="yii2", help="Dataset name")
    args = parser.parse_args()
    
    model, metrics = main(args.dataset)
