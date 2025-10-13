# 快速参考指南 (Quick Reference Guide)

## 常用命令 (Common Commands)

### 运行模型 (Run Models)

```powershell
# Lab1 - 回归任务
python main.py --lab 1 --model linear --task 1 --dataset yii2

# Lab1 - 分类任务
python main.py --lab 1 --model logistic --task 2 --dataset django

# Lab2 - 深度学习回归
python main.py --lab 2 --model mlp --task 1 --dataset tensorflow

# 使用原始数据 (不使用预提取特征)
python main.py --lab 2 --model mlp --task 1 --dataset yii2 --no-extracted
```

### 查看帮助 (Get Help)

```powershell
# 查看所有参数
python main.py --help

# 查看具体模型的参数
python models/lab1/task1_linear_refactored.py --help
```

## 代码模板 (Code Templates)

### 添加新的 Lab1 模型

```python
"""
Your Model Description
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.your_model import YourModel
from config import get_data_path, LAB1_CONFIG, RANDOM_SEED, TRAIN_SPLIT_RATIO
from utils.data_loader import load_and_merge_data, prepare_features, train_test_split_by_time
from models.lab1.base_model import Lab1BaseModel


class YourModelClass(Lab1BaseModel):
    """Your model description"""
    
    def __init__(self, config=None, random_seed=RANDOM_SEED):
        super().__init__(
            model_type="your_model",
            task="regression",  # or "classification"
            config=config or LAB1_CONFIG["your_model"],
            random_seed=random_seed
        )
    
    def build_model(self):
        """Build your model"""
        self.model = YourModel(**self.config)
        return self.model


def main(dataset_name="yii2"):
    print("="*60)
    print("Your Task Description")
    print("="*60)
    
    # 1. Load data
    data_path = get_data_path(dataset_name)
    df = load_and_merge_data(data_path)
    
    # 2. Prepare features
    X, y, features = prepare_features(
        df,
        task="regression",  # or "classification"
        apply_log_transform=True
    )
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split_by_time(
        X, y, df, split_ratio=TRAIN_SPLIT_RATIO
    )
    
    # 4. Initialize and build model
    model = YourModelClass()
    model.build_model()
    
    # 5. Preprocess
    X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test)
    
    # 6. Train
    model.train(X_train_scaled, y_train)
    
    # 7. Evaluate
    metrics = model.evaluate(X_test_scaled, y_test, verbose=True)
    
    # 8. Analyze
    model.analyze_feature_importance(top_n=10)
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Your Model")
    parser.add_argument("--dataset", type=str, default="yii2")
    args = parser.parse_args()
    model, metrics = main(args.dataset)
```

### 添加新的 Lab2 模型

```python
"""
Your Deep Learning Model Description
"""
import sys
from pathlib import Path
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


class YourNetwork(nn.Module):
    """Your neural network architecture"""
    
    def __init__(self, input_dim, hidden_layers=[128, 64], dropout_rate=0.2):
        super(YourNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class YourDLModel(Lab2BaseModel):
    """Your deep learning model class"""
    
    def __init__(self, config=None, random_seed=RANDOM_SEED):
        training_config = LAB2_CONFIG.get("training", {})
        if config:
            training_config.update(config)
        
        super().__init__(
            model_name="your_model",
            task="regression",
            config=training_config,
            random_seed=random_seed
        )
        
        self.model_config = LAB2_CONFIG.get("your_model", {})
    
    def build_model(self, input_dim: int):
        """Build the neural network"""
        self.model = YourNetwork(
            input_dim=input_dim,
            hidden_layers=self.model_config.get("hidden_layers", [128, 64]),
            dropout_rate=self.model_config.get("dropout_rate", 0.2)
        ).to(self.device)
        
        print(f"\nModel Architecture:\n{self.model}")
        return self.model


def main(dataset_name="yii2", use_extracted=True):
    print("="*60)
    print("Your Deep Learning Task")
    print("="*60)
    
    # 1. Load data
    data_path = get_data_path(dataset_name)
    df = load_and_merge_data(data_path, use_extracted=use_extracted)
    
    # 2. Prepare features
    X, y, features = prepare_features(df, task="regression")
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split_by_time(X, y, df)
    
    # 4. Initialize model
    model = YourDLModel()
    
    # 5. Preprocess
    X_train_scaled, X_test_scaled = model.preprocess_data(X_train, X_test)
    
    # 6. Build model
    model.build_model(X_train_scaled.shape[1])
    
    # 7. Prepare DataLoaders
    batch_size = model.config.get("batch_size", 64)
    train_loader, test_loader = model.prepare_dataloaders(
        X_train_scaled, y_train, X_test_scaled, y_test, batch_size
    )
    
    # 8. Train
    model.train(
        train_loader, test_loader,
        n_epochs=model.config.get("n_epochs", 100),
        learning_rate=model.config.get("learning_rate", 0.001),
        patience=model.config.get("patience", 10)
    )
    
    # 9. Evaluate
    metrics = model.evaluate(test_loader, y_test, verbose=True)
    
    # 10. Save
    save_path = get_checkpoint_path("your_model", dataset_name)
    model.save_model(save_path)
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Your DL Model")
    parser.add_argument("--dataset", type=str, default="yii2")
    parser.add_argument("--no-extracted", action="store_true")
    args = parser.parse_args()
    model, metrics = main(args.dataset, use_extracted=not args.no_extracted)
```

## 配置修改 (Configuration Changes)

### 修改 config.py

```python
# 添加新模型配置
LAB1_CONFIG = {
    # ... 现有配置 ...
    "your_new_model": {
        "param1": value1,
        "param2": value2,
        "random_state": RANDOM_SEED,
    }
}

LAB2_CONFIG = {
    # ... 现有配置 ...
    "your_new_model": {
        "hidden_layers": [256, 128, 64],
        "dropout_rate": 0.3,
    }
}
```

### 在 main.py 中注册新模型

```python
# 在 run_lab1_model() 或 run_lab2_model() 中添加:
elif model_type == "your_model":
    from models.lab1.your_model_refactored import main
    print(f"\nRunning Lab1: Your Model on {dataset}")
    return main(dataset)
```

## 工具函数使用 (Utility Functions)

### 数据加载

```python
from utils.data_loader import load_and_merge_data

# 加载预处理的数据
df = load_and_merge_data(data_path, use_extracted=True)

# 加载原始数据
df = load_and_merge_data(data_path, use_extracted=False)
```

### 特征准备

```python
from utils.data_loader import prepare_features

# 回归任务
X, y, features = prepare_features(
    df,
    task="regression",
    apply_log_transform=True,
    max_ttc_hours=1000
)

# 分类任务
X, y, features = prepare_features(
    df,
    task="classification",
    target_col="merged"
)
```

### 时序分割

```python
from utils.data_loader import train_test_split_by_time

X_train, X_test, y_train, y_test = train_test_split_by_time(
    X, y, df, split_ratio=0.8
)
```

## 调试技巧 (Debugging Tips)

### 1. 测试数据加载

```python
from config import get_data_path
from utils.data_loader import load_and_merge_data

data_path = get_data_path("yii2")
df = load_and_merge_data(data_path)
print(df.head())
print(df.info())
```

### 2. 检查模型配置

```python
from config import LAB1_CONFIG, LAB2_CONFIG

print("Lab1 Config:", LAB1_CONFIG)
print("Lab2 Config:", LAB2_CONFIG)
```

### 3. 测试模型初始化

```python
from models.lab1.task1_linear_refactored import LinearRegressionModel

model = LinearRegressionModel()
model.build_model()
print(model.model)
```

## 常见问题 (FAQ)

### Q: 如何更改训练参数？
A: 修改 `config.py` 中的相应配置，或在实例化模型时传入自定义配置。

### Q: 如何添加新的数据集？
A: 将数据文件放入 `data/<dataset_name>/` 目录，然后在 `config.py` 的 `DATASETS` 列表中添加名称。

### Q: 如何保存和加载模型？
A: 
```python
# 保存
from config import get_checkpoint_path
save_path = get_checkpoint_path("model_name", "dataset_name")
model.save_model(save_path)

# 加载
model.load_model(save_path)
```

### Q: 原始代码文件怎么办？
A: 保留原始文件作为参考，使用新的 `_refactored.py` 文件。

### Q: 如何切换 GPU/CPU？
A: 在 `config.py` 中修改 `LAB2_CONFIG["training"]["device"]` 为 "cuda" 或 "cpu"。

## 性能优化建议 (Performance Tips)

1. **使用预提取特征**: Lab2 模型运行时使用 `--no-extracted` 会更快
2. **批量大小**: 根据 GPU 内存调整 `batch_size`
3. **早停参数**: 调整 `patience` 避免过拟合
4. **数据缓存**: 大数据集可以先运行 `utils/feature_extract.py` 生成预处理文件

## 更多帮助 (More Help)

- 查看 `README.md` 了解详细文档
- 查看 `REFACTORING_SUMMARY.md` 了解重构详情
- 运行 `python <script> --help` 查看参数说明
