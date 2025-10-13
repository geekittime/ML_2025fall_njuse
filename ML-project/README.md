# ML-Project: Pull Request Analysis and Prediction

A refactored and well-structured machine learning project for analyzing and predicting GitHub Pull Request metrics using both classical ML and deep learning approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Models](#models)
- [Data](#data)
- [Development](#development)

## 🎯 Overview

This project implements machine learning models to predict two key Pull Request (PR) metrics:

### Task 1: PR Time-to-Close Prediction (Regression)
Predict how long it will take for a PR to be closed based on various features like PR size, author history, review patterns, etc.

### Task 2: PR Merge Status Prediction (Classification)
Predict whether a PR will be merged or rejected based on similar features.

The project includes:
- **Lab1**: Classical ML models (Linear Regression, Logistic Regression, Random Forest, etc.)
- **Lab2**: Deep Learning models (MLP, Wide&Deep, DeepCross, Multitask Learning)

## 📁 Project Structure

```
ML-project/
├── config.py                 # Centralized configuration
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── data/                    # Datasets for different projects
│   ├── yii2/
│   ├── django/
│   ├── tensorflow/
│   └── ...
│
├── models/                  # Model implementations
│   ├── __init__.py
│   ├── lab1/               # Classical ML models
│   │   ├── __init__.py
│   │   ├── base_model.py                    # Base class for Lab1
│   │   ├── task1_linear_refactored.py       # Linear Regression
│   │   ├── task2_Logistic_refactored.py     # Logistic Regression
│   │   └── ...
│   └── lab2/               # Deep Learning models
│       ├── __init__.py
│       ├── base_model.py                    # Base class for Lab2
│       ├── task1_Linear_refactored.py       # MLP Regressor
│       └── ...
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── feature_extract.py  # Feature extraction
│   └── ...
│
└── checkpoints/            # Saved model checkpoints
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository** (or navigate to the project directory)
```powershell
cd "c:\Users\COOL_TEA\Course\DeepLearning\Lab\ML-project"
```

2. **Install dependencies**
```powershell
pip install -r requirements.txt
```

The main dependencies include:
- pandas, numpy - Data manipulation
- scikit-learn - Classical ML models
- torch, torchvision - Deep learning
- openpyxl - Excel file handling

## 🚀 Quick Start

### Using the Main Entry Point

The easiest way to run models is through `main.py`:

```powershell
# Run Lab1 Linear Regression on yii2 dataset
python main.py --lab 1 --model linear --task 1 --dataset yii2

# Run Lab1 Logistic Regression for classification
python main.py --lab 1 --model logistic --task 2 --dataset yii2

# Run Lab2 MLP (Deep Learning)
python main.py --lab 2 --model mlp --task 1 --dataset yii2
```

### Running Individual Model Files

You can also run model files directly:

```powershell
# Lab1 models
python models/lab1/task1_linear_refactored.py --dataset yii2
python models/lab1/task2_Logistic_refactored.py --dataset django

# Lab2 models
python models/lab2/task1_Linear_refactored.py --dataset tensorflow
```

## ⚙️ Configuration

All configuration is centralized in `config.py`. Key configuration areas:

### Paths
```python
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
```

### Model Hyperparameters
```python
LAB1_CONFIG = {
    "linear_regression": {
        "alpha": 1.0,
        "max_iter": 1000,
    },
    "logistic_regression": {
        "class_weight": "balanced",
        "max_iter": 3000,
    }
}

LAB2_CONFIG = {
    "training": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "n_epochs": 100,
        "patience": 10,
    }
}
```

### Data Processing
```python
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42
FEATURE_CONFIG = {
    "apply_log_transform": True,
    "max_ttc_hours": 1000,
}
```

## 📖 Usage

### Command Line Arguments

#### main.py Options

```
--lab {1,2}          Lab number: 1 (Classical ML) or 2 (Deep Learning)
--model MODEL        Model type (see below)
--task {1,2}         Task: 1 (Regression) or 2 (Classification)
--dataset DATASET    Dataset name (yii2, django, tensorflow, etc.)
--no-extracted       Don't use pre-extracted features (Lab2 only)
```

#### Lab1 Models
- Task 1: `linear`, `polynomial`, `random_forest`
- Task 2: `logistic`, `random_forest`

#### Lab2 Models
- Task 1: `mlp`, `wide_deep`, `deepcross`
- Task 2: `mlp`, `wide_deep`, `deepcross`

### Examples

```powershell
# Run different models on different datasets
python main.py --lab 1 --model linear --task 1 --dataset yii2
python main.py --lab 1 --model logistic --task 2 --dataset django
python main.py --lab 2 --model mlp --task 1 --dataset tensorflow

# Use raw data instead of extracted features (Lab2)
python main.py --lab 2 --model mlp --task 1 --dataset yii2 --no-extracted
```

## 🤖 Models

### Lab1: Classical Machine Learning

#### Base Class: `Lab1BaseModel`
- Provides unified interface for all Lab1 models
- Handles data preprocessing, standardization
- Automatic evaluation metrics (MAE, RMSE, R², Accuracy)
- Feature importance analysis

#### Implemented Models
1. **Linear Regression** (Ridge) - Task 1
   - Time-to-close prediction
   - L2 regularization

2. **Logistic Regression** - Task 2
   - Merge status classification
   - Balanced class weights

3. **Polynomial Regression** - Task 1 *(to be refactored)*
4. **Random Forest** - Task 1 & 2 *(to be refactored)*

### Lab2: Deep Learning (PyTorch)

#### Base Class: `Lab2BaseModel`
- Unified interface for PyTorch models
- Automatic GPU/CPU device selection
- Early stopping mechanism
- Model checkpointing
- Training history tracking

#### Implemented Models
1. **MLP (Multi-Layer Perceptron)** - Task 1
   - 3-layer neural network
   - ReLU activation, Dropout regularization
   - MSE loss for regression

2. **Wide & Deep** - Task 1 & 2 *(to be refactored)*
3. **DeepCross** - Task 1 & 2 *(to be refactored)*
4. **Multitask Learning** *(to be refactored)*

## 📊 Data

### Available Datasets
- yii2
- django
- moby
- opencv
- react
- salt
- scikit-learn
- symfony
- tensorflow
- terraform

### Data Files
Each dataset folder contains:
- `PR_info.xlsx` - Basic PR information
- `PR_features.xlsx` - Extracted PR features
- `author_features.xlsx` - Author-related features
- `PR_info_add_conversation.xlsx` - Conversation features
- `PR_extracted_features.xlsx` - Pre-processed combined features

### Feature Engineering

The `utils/data_loader.py` module provides:
- Automatic data loading and merging
- Feature preparation for regression/classification
- Temporal train/test splitting
- Missing value handling
- Log transformation for target variables

## 🛠️ Development

### Adding a New Model

1. **For Lab1 (Classical ML)**:

```python
from models.lab1.base_model import Lab1BaseModel
from sklearn.ensemble import GradientBoostingRegressor

class MyNewModel(Lab1BaseModel):
    def __init__(self, config=None):
        super().__init__(
            model_type="my_model",
            task="regression",
            config=config
        )
    
    def build_model(self):
        self.model = GradientBoostingRegressor(**self.config)
        return self.model
```

2. **For Lab2 (Deep Learning)**:

```python
import torch.nn as nn
from models.lab2.base_model import Lab2BaseModel

class MyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(...)
    
    def forward(self, x):
        return self.layers(x)

class MyNewDLModel(Lab2BaseModel):
    def build_model(self, input_dim):
        self.model = MyNetwork(input_dim).to(self.device)
        return self.model
```

### Testing Utilities

```powershell
# Test configuration
python config.py

# Test data loading
python utils/data_loader.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Document functions with docstrings
- Keep functions focused and modular

## 📝 Notes

### Migration from Old Code
- Old script-style files (e.g., `task1_linear.py`) are preserved
- New refactored versions have `_refactored.py` suffix
- Both can coexist during transition

### Best Practices
1. Always use `config.py` for configuration
2. Use base classes for new models
3. Leverage utility functions in `utils/`
4. Save models to `checkpoints/` directory
5. Use temporal splitting for time-series data

## 🤝 Contributing

When adding new features:
1. Update `config.py` if adding new hyperparameters
2. Extend base classes if needed
3. Update this README
4. Follow the existing code structure

## 📄 License

This is an academic project for educational purposes.

---

**Author**: ML-Project Team  
**Last Updated**: 2025-10  
**Version**: 2.0 (Refactored)