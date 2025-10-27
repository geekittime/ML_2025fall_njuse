"""
Base model class for Lab2 deep learning models using PyTorch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)
import warnings

warnings.filterwarnings('ignore')


class Lab2BaseModel:
    """Base class for Lab2 deep learning models with PyTorch"""
    
    def __init__(
        self,
        model_name: str,
        task: str = "regression",
        config: Dict[str, Any] = None,
        random_seed: int = 42
    ):
        """
        Initialize Lab2 base model
        
        Args:
            model_name: Name of the model (e.g., "linear_mlp", "wide_deep")
            task: "regression" or "classification"
            config: Model and training configuration
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.task = task
        self.config = config or {}
        self.random_seed = random_seed
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Device configuration
        device_config = self.config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_config)
        
        print(f"Using device: {self.device}")
        
        # Model and training components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.optimizer = None
        self.criterion = None
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def build_model(self, input_dim: int) -> nn.Module:
        """Build the neural network model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def preprocess_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess and standardize data
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        self.feature_names = X_train.columns.tolist()
        
        print("Applying feature standardization...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def prepare_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
        batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare PyTorch DataLoaders
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            batch_size: Batch size for training
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Convert to tensors and move to device
        X_train_tensor = torch.tensor(X_train.astype(np.float32)).to(self.device)
        y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).to(self.device).view(-1, 1)
        X_test_tensor = torch.tensor(X_test.astype(np.float32)).to(self.device)
        y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).to(self.device).view(-1, 1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Maximum number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            verbose: Whether to print progress
        
        Returns:
            Training history dictionary
        """
        # Setup optimizer and loss
        if self.task == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        print(f"\nTraining {self.model_name} model...")
        
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            
            # Store history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            # Print progress
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
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
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions
        
        Args:
            data_loader: DataLoader with input data
        
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                outputs = self.model(inputs)
                if self.task == "classification":
                    outputs = torch.sigmoid(outputs)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions).flatten()
    
    def evaluate(
        self,
        test_loader: DataLoader,
        y_test: pd.Series,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            test_loader: Test data loader
            y_test: True test labels
            verbose: Whether to print results
        
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(test_loader)
        
        if self.task == "regression":
            return self._evaluate_regression(y_test, predictions, verbose)
        else:
            return self._evaluate_classification(y_test, predictions, verbose)
    
    def _evaluate_regression(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate regression model"""
        # Transform back from log space
        y_true_original = np.expm1(y_true)
        y_pred_original = np.expm1(y_pred)
        
        mae = mean_absolute_error(y_true_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
        r2 = r2_score(y_true_original, y_pred_original)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        if verbose:
            print("\n" + "="*60)
            print("Deep Learning Model Evaluation Results (Regression)")
            print("="*60)
            print(f"Mean Absolute Error (MAE): {mae:.2f} hours ({mae/24:.2f} days)")
            print(f"Root Mean Squared Error (RMSE): {rmse:.2f} hours ({rmse/24:.2f} days)")
            print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
            print("="*60)
        
        return metrics
    
    def _evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred_prob: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate classification model"""
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
        
        if verbose:
            print("\n" + "="*60)
            print("Deep Learning Model Evaluation Results (Classification)")
            print("="*60)
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print("\nConfusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=['Not Merged', 'Merged']))
            print("="*60)
        
        return metrics
    
    def save_model(self, save_path: Path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path, model: nn.Module):
        """Load a trained model"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model = model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.config = checkpoint.get('config', {})
        self.feature_names = checkpoint.get('feature_names', None)
        self.training_history = checkpoint.get('training_history', {})
        print(f"Model loaded from {load_path}")
