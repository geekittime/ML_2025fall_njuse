"""
Base model class for Lab1 classical machine learning models
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)
import warnings

warnings.filterwarnings('ignore')


class Lab1BaseModel:
    """Base class for Lab1 classical ML models"""
    
    def __init__(
        self,
        model_type: str,
        task: str = "regression",
        config: Dict[str, Any] = None,
        random_seed: int = 42
    ):
        """
        Initialize Lab1 base model
        
        Args:
            model_type: Type of model (e.g., "linear", "polynomial", "random_forest")
            task: "regression" or "classification"
            config: Model-specific configuration
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.task = task
        self.config = config or {}
        self.random_seed = random_seed
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        np.random.seed(random_seed)
    
    def build_model(self):
        """Build the model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def preprocess_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess training and test data
        
        Args:
            X_train: Training features
            X_test: Test features
            scale: Whether to apply StandardScaler
        
        Returns:
            Tuple of (X_train_processed, X_test_processed)
        """
        self.feature_names = X_train.columns.tolist()
        
        if scale:
            print("Applying feature standardization...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        else:
            return X_train.values, X_test.values
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: pd.Series
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        print(f"\nTraining {self.model_type} model for {self.task}...")
        self.model.fit(X_train, y_train)
        print("Model training completed.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
        
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: pd.Series,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Whether to print results
        
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        
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
        # Transform back from log space if needed
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
            print("\n" + "="*50)
            print("Regression Model Evaluation Results")
            print("="*50)
            print(f"Mean Absolute Error (MAE): {mae:.2f} hours ({mae/24:.2f} days)")
            print(f"Root Mean Squared Error (RMSE): {rmse:.2f} hours ({rmse/24:.2f} days)")
            print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
            print("="*50)
        
        return metrics
    
    def _evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate classification model"""
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
        
        if verbose:
            print("\n" + "="*50)
            print("Classification Model Evaluation Results")
            print("="*50)
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print("\nConfusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=['Not Merged', 'Merged']))
            print("="*50)
        
        return metrics
    
    def analyze_feature_importance(self, top_n: int = 10):
        """
        Analyze and display feature importance/coefficients
        
        Args:
            top_n: Number of top features to display
        """
        if self.feature_names is None:
            print("No feature names available. Train the model first.")
            return
        
        if hasattr(self.model, 'coef_'):
            # Linear models
            coef = self.model.coef_
            if len(coef.shape) > 1:
                coef = coef[0]
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coef,
                'abs_coefficient': np.abs(coef)
            })
            feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
            
            print(f"\n{'='*50}")
            print(f"Top {top_n} Features by Coefficient Magnitude")
            print(f"{'='*50}")
            print("Note: Positive = increases target, Negative = decreases target")
            print(feature_importance.head(top_n).to_string(index=False))
            print(f"{'='*50}\n")
            
        elif hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            print(f"\n{'='*50}")
            print(f"Top {top_n} Features by Importance")
            print(f"{'='*50}")
            print(feature_importance.head(top_n).to_string(index=False))
            print(f"{'='*50}\n")
        else:
            print(f"Feature importance not available for {self.model_type} model")
    
    def save_model(self, save_path: Path):
        """Save the trained model (can be extended with joblib if needed)"""
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Path):
        """Load a trained model"""
        import joblib
        data = joblib.load(load_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.config = data['config']
        print(f"Model loaded from {load_path}")
