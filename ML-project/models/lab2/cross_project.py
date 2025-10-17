"""
Cross-Project Prediction Experiments
Train on one project, test on another to evaluate model generalization
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import List, Dict
from config import DATASETS, get_data_path, RANDOM_SEED
from models.lab2.task1 import Task1RegressionModel
from models.lab2.task2 import Task2ClassificationModel
from utils.data_loader import load_and_merge_data, prepare_features


class CrossProjectExperiment:
    """Cross-project prediction experiment manager"""
    
    def __init__(self, model_type="mlp", task="regression", random_seed=RANDOM_SEED):
        """
        Args:
            model_type: Type of model to use
            task: "regression" (Task 1) or "classification" (Task 2)
            random_seed: Random seed
        """
        self.model_type = model_type
        self.task = task
        self.random_seed = random_seed
        self.results = []
    
    def run_single_experiment(self, train_dataset: str, test_dataset: str):
        """
        Run single cross-project experiment
        
        Args:
            train_dataset: Dataset to train on
            test_dataset: Dataset to test on
        
        Returns:
            Dictionary containing experiment results
        """
        print("\n" + "="*70)
        print(f"Cross-Project Experiment: Train on {train_dataset}, Test on {test_dataset}")
        print("="*70)
        
        # 1. Load training data
        train_path = get_data_path(train_dataset)
        train_df = load_and_merge_data(train_path, use_extracted=True)
        
        # 2. Load test data
        test_path = get_data_path(test_dataset)
        test_df = load_and_merge_data(test_path, use_extracted=True)
        
        # 3. Prepare features
        X_train, y_train, train_features, _ = prepare_features(
            train_df,
            task=self.task,
            apply_log_transform=(self.task == "regression"),
            max_ttc_hours=1000 if self.task == "regression" else None
        )
        
        X_test, y_test, test_features, _ = prepare_features(
            test_df,
            task=self.task,
            apply_log_transform=(self.task == "regression"),
            max_ttc_hours=1000 if self.task == "regression" else None
        )
        
        # 4. Align features (use intersection of features from both datasets)
        common_features = list(set(train_features) & set(test_features))
        print(f"\nFeature alignment:")
        print(f"  Train features: {len(train_features)}")
        print(f"  Test features: {len(test_features)}")
        print(f"  Common features: {len(common_features)}")
        
        X_train = X_train[common_features]
        X_test = X_test[common_features]
        
        # 5. Initialize and build model
        if self.task == "regression":
            model = Task1RegressionModel(model_type=self.model_type, random_seed=self.random_seed)
        else:
            model = Task2ClassificationModel(model_type=self.model_type, random_seed=self.random_seed)
        
        # 6. Preprocess data
        X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test)
        
        # 7. Build model
        input_dim = X_train_scaled.shape[1]
        model.build_model(input_dim)
        
        # 8. Prepare DataLoaders
        train_loader, _ = model.prepare_dataloaders(
            X_train_scaled, y_train,
            X_train_scaled[:100], y_train[:100],  # Dummy val set
            batch_size=64
        )
        
        _, test_loader = model.prepare_dataloaders(
            X_test_scaled[:100], y_test[:100],  # Dummy train set
            X_test_scaled, y_test,
            batch_size=64
        )
        
        # 9. Train model
        print(f"\nTraining {self.model_type.upper()} on {train_dataset}...")
        model.train(
            train_loader=train_loader,
            val_loader=train_loader,  # Use train as val for cross-project
            n_epochs=50,  # Fewer epochs for cross-project
            learning_rate=0.001,
            patience=10,
            verbose=False
        )
        
        # 10. Evaluate on test dataset
        print(f"\nEvaluating on {test_dataset}...")
        metrics = model.evaluate(test_loader, y_test, verbose=True)
        
        # 11. Store results
        result = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'model_type': self.model_type,
            'task': self.task,
            'metrics': metrics,
            'common_features': len(common_features)
        }
        
        self.results.append(result)
        
        return result
    
    def run_full_matrix(self, datasets: List[str] = None):
        """
        Run full cross-project matrix (train on each, test on all others)
        
        Args:
            datasets: List of datasets to use (if None, use all available)
        """
        if datasets is None:
            # Use datasets that have data
            datasets = ["yii2", "django", "tensorflow"]  # Add more as available
        
        print("\n" + "="*70)
        print(f"Full Cross-Project Experiment Matrix")
        print(f"Model: {self.model_type.upper()}, Task: {self.task}")
        print(f"Datasets: {', '.join(datasets)}")
        print("="*70)
        
        for train_ds in datasets:
            for test_ds in datasets:
                if train_ds != test_ds:
                    try:
                        self.run_single_experiment(train_ds, test_ds)
                    except Exception as e:
                        print(f"\n❌ Error in {train_ds} → {test_ds}: {e}")
                        continue
        
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of all experiments"""
        if not self.results:
            print("\nNo results to summarize.")
            return
        
        print("\n" + "="*70)
        print("Cross-Project Experiment Summary")
        print("="*70)
        
        # Create results table
        summary_data = []
        for result in self.results:
            row = {
                'Train': result['train_dataset'],
                'Test': result['test_dataset'],
                'Model': result['model_type']
            }
            
            if self.task == "regression":
                row['MAE'] = f"{result['metrics']['mae']:.2f}"
                row['RMSE'] = f"{result['metrics']['rmse']:.2f}"
                row['R²'] = f"{result['metrics']['r2']:.4f}"
            else:
                row['Accuracy'] = f"{result['metrics']['accuracy']:.4f}"
                row['Precision'] = f"{result['metrics']['precision']:.4f}"
                row['Recall'] = f"{result['metrics']['recall']:.4f}"
                row['F1'] = f"{result['metrics']['f1_macro']:.4f}"
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print("\n", summary_df.to_string(index=False))
        print("="*70)
        
        # Save to CSV
        output_path = project_root / f"cross_project_results_{self.task}_{self.model_type}.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


def main(model_type="mlp", task="regression", train_dataset=None, test_dataset=None, full_matrix=False):
    """
    Main function for cross-project experiments
    
    Args:
        model_type: Model type to use
        task: "regression" or "classification"
        train_dataset: Training dataset (if single experiment)
        test_dataset: Test dataset (if single experiment)
        full_matrix: Whether to run full cross-project matrix
    """
    experiment = CrossProjectExperiment(model_type=model_type, task=task)
    
    if full_matrix:
        # Run full matrix
        available_datasets = ["yii2"]  # Add more as you have data
        experiment.run_full_matrix(available_datasets)
    elif train_dataset and test_dataset:
        # Run single experiment
        experiment.run_single_experiment(train_dataset, test_dataset)
    else:
        print("Error: Please specify either --full-matrix or both --train and --test datasets")
        return None
    
    return experiment


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cross-Project Prediction Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single cross-project experiment
  python cross_project.py --model mlp --task regression --train yii2 --test django
  
  # Full cross-project matrix
  python cross_project.py --model wide_deep --task classification --full-matrix
  
  # Classification task
  python cross_project.py --model deep_cross --task classification --train tensorflow --test yii2
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "wide_deep", "deep_cross"],
        help="Model architecture"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type"
    )
    
    parser.add_argument(
        "--train",
        type=str,
        help="Training dataset name"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        help="Test dataset name"
    )
    
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Run full cross-project matrix"
    )
    
    args = parser.parse_args()
    
    experiment = main(
        model_type=args.model,
        task=args.task,
        train_dataset=args.train,
        test_dataset=args.test,
        full_matrix=args.full_matrix
    )
