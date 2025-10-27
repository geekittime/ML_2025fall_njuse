"""
ML-Project Main Entry Point
Unified interface for running Lab1 and Lab2 models

Usage:
    python main.py --lab 1 --model linear --task 1 --dataset yii2
    python main.py --lab 2 --model mlp --task 1 --dataset yii2
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import DATASETS


def run_lab1_model(model_type: str, task: int, dataset: str):
    """
    Run a Lab1 (classical ML) model
    
    Args:
        model_type: Type of model ('linear', 'polynomial', 'random_forest')
        task: Task number (1 or 2)
        dataset: Dataset name
    """
    return
    if task == 1:
        # Task 1: Regression - Predicting PR Time-to-Close
        if model_type == "linear":
            from models.lab1.task1_linear_refactored import main
            print(f"\nRunning Lab1 Task1: Linear Regression on {dataset}")
            return main(dataset)
        elif model_type == "polynomial":
            print("Polynomial regression not yet refactored. Please refactor task1_poly.py")
            # from models.lab1.task1_poly_refactored import main
            # return main(dataset)
        elif model_type == "random_forest":
            print("Random Forest not yet refactored. Please refactor task1_forest.py")
            # from models.lab1.task1_forest_refactored import main
            # return main(dataset)
        else:
            raise ValueError(f"Unknown Lab1 Task1 model: {model_type}")
    
    elif task == 2:
        # Task 2: Classification - Predicting PR Merge Status
        if model_type == "logistic":
            from models.lab1.task2_Logistic_refactored import main
            print(f"\nRunning Lab1 Task2: Logistic Regression on {dataset}")
            return main(dataset)
        elif model_type == "random_forest":
            print("Random Forest classification not yet refactored. Please refactor task2_forest.py")
            # from models.lab1.task2_forest_refactored import main
            # return main(dataset)
        else:
            raise ValueError(f"Unknown Lab1 Task2 model: {model_type}")
    
    else:
        raise ValueError(f"Unknown task number: {task}. Must be 1 or 2")


def run_lab2_model(model_type: str, task, dataset: str, use_extracted: bool = True):
    """
    Run a Lab2 (deep learning) model
    
    Args:
        model_type: Type of model ('mlp', 'wide_deep', 'deep_cross', 'shared_bottom', 'mmoe', 'multitask')
        task: Task number (1, 2, or 'multitask')
        dataset: Dataset name
        use_extracted: Whether to use pre-extracted features
    """
    if task == 1:
        # Task 1: Regression - Predicting PR Time-to-Close
        from models.lab2.task1 import main
        print(f"\nRunning Lab2 Task1: {model_type.upper()} on {dataset}")
        return main(model_type, dataset, use_extracted)
    
    elif task == 2:
        # Task 2: Classification - Predicting PR Merge Status
        from models.lab2.task2 import main
        print(f"\nRunning Lab2 Task2: {model_type.upper()} on {dataset}")
        return main(model_type, dataset, use_extracted)
    
    elif task == "multitask":
        # Multi-Task Learning
        from models.lab2.multitask import main
        print(f"\nRunning Lab2 Multi-Task Learning on {dataset}")
        return main(dataset)
    
    else:
        raise ValueError(f"Unknown task: {task}. Must be 1, 2, or 'multitask'")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ML-Project: Unified interface for Lab1 and Lab2 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lab1: Classical Machine Learning
  python main.py --lab 1 --model linear --task 1 --dataset yii2
  python main.py --lab 1 --model logistic --task 2 --dataset django
  
  # Lab2 Task1: Regression (PR Time-to-Close Prediction)
  python main.py --lab 2 --model mlp --task 1 --dataset yii2
  python main.py --lab 2 --model wide_deep --task 1 --dataset tensorflow
  python main.py --lab 2 --model deep_cross --task 1 --dataset django
  python main.py --lab 2 --model shared_bottom --task 1 --dataset yii2
  python main.py --lab 2 --model mmoe --task 1 --dataset yii2
  
  # Lab2 Task2: Classification (PR Merge Prediction)
  python main.py --lab 2 --model mlp --task 2 --dataset yii2
  python main.py --lab 2 --model wide_deep --task 2 --dataset django
  python main.py --lab 2 --model deep_cross --task 2 --dataset tensorflow
  
  # Lab2: Multi-Task Learning (both tasks together)
  python main.py --lab 2 --model - --task 1 --dataset yii2
        """
    )
    
    parser.add_argument(
        "--lab",
        type=int,
        required=True,
        choices=[1, 2],
        help="Lab number: 1 (Classical ML) or 2 (Deep Learning)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model type. Lab1: linear, polynomial, random_forest, logistic. Lab2: mlp, wide_deep, deep_cross, shared_bottom, mmoe"
    )
    
    parser.add_argument(
        "--task",
        required=True,
        choices=[1, 2, "multitask"],
        help="Task number: 1 (Regression - TTC Prediction), 2 (Classification - Merge Prediction) or multitask (Lab2 only)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="yii2",
        choices=DATASETS,
        help=f"Dataset name. Available: {', '.join(DATASETS)}"
    )
    
    parser.add_argument(
        "--no-extracted",
        action="store_true",
        help="Don't use pre-extracted features (Lab2 only)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("ML-Project Execution Configuration")
    print("="*70)
    print(f"Lab: {args.lab} ({'Classical ML' if args.lab == 1 else 'Deep Learning'})")
    print(f"Model: {args.model}")
    print(f"Task: {args.task} ({'Regression' if args.task == 1 else 'Classification'})")
    print(f"Dataset: {args.dataset}")
    if args.lab == 2:
        print(f"Use Extracted Features: {not args.no_extracted}")
    print("="*70)
    
    # Run the appropriate model
    try:
        if args.lab == 1:
            model, metrics = run_lab1_model(args.model, args.task, args.dataset)
        else:  # lab == 2
            model, metrics = run_lab2_model(args.model, args.task, args.dataset, 
                                           use_extracted=not args.no_extracted)
        
        print("\n" + "="*70)
        print("Execution Completed Successfully!")
        print("="*70)
        print(f"Final Metrics: {metrics}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
