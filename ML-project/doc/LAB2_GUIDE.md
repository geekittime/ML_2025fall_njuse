# Lab2 深度学习模型使用指南

## 概述

Lab2 实现了多种深度学习模型用于 Pull Request 分析：

### 任务说明

1. **Task 1 (回归)**: 预测 PR 关闭时长 (Time-to-Close)
   - 目标: 预测 PR 从创建到关闭需要多少小时
   - 评价指标: MAE, MSE, RMSE, R²

2. **Task 2 (分类)**: 预测 PR 是否会被合入
   - 目标: 预测 PR 最终是否会被 merge
   - 评价指标: Accuracy, Precision (Macro), Recall (Macro), F1 Score (Macro)

3. **Multi-Task Learning**: 同时预测两个任务
   - 共享底层特征表示
   - 同时优化两个任务的损失

## 支持的模型架构

### Task 1 (回归) 模型

| 模型              | 说明           | 特点                       |
| ----------------- | -------------- | -------------------------- |
| **MLP**           | 多层感知机     | 简单的全连接神经网络       |
| **Wide & Deep**   | 宽度+深度网络  | 结合记忆能力和泛化能力     |
| **Deep & Cross**  | 深度交叉网络   | 显式学习特征交叉           |
| **Shared-Bottom** | 共享底层网络   | 多任务学习架构(单任务模式) |
| **MMoE**          | 多门控专家混合 | 多个专家网络+门控机制      |

### Task 2 (分类) 模型

| 模型             | 说明              |
| ---------------- | ----------------- |
| **MLP**          | 多层感知机分类器  |
| **Wide & Deep**  | 宽度+深度分类网络 |
| **Deep & Cross** | 深度交叉分类网络  |

### Multi-Task 模型

| 模型          | 说明                       |
| ------------- | -------------------------- |
| **MultiTask** | 同时预测 TTC 和 Merge 状态 |

## 快速开始

### 1. Task 1: 回归任务

```powershell
# MLP
python models/lab2/task1.py --model mlp --dataset yii2

# Wide & Deep
python models/lab2/task1.py --model wide_deep --dataset yii2

# Deep & Cross
python models/lab2/task1.py --model deep_cross --dataset yii2

# Shared-Bottom
python models/lab2/task1.py --model shared_bottom --dataset yii2

# MMoE
python models/lab2/task1.py --model mmoe --dataset yii2
```

### 2. Task 2: 分类任务

```powershell
# MLP
python models/lab2/task2.py --model mlp --dataset yii2

# Wide & Deep
python models/lab2/task2.py --model wide_deep --dataset yii2

# Deep & Cross
python models/lab2/task2.py --model deep_cross --dataset yii2
```

### 3. Multi-Task Learning

```powershell
# 多任务学习
python models/lab2/multitask.py --dataset yii2
```

### 4. 通过 main.py 运行

```powershell
# Task 1
python main.py --lab 2 --model mlp --task 1 --dataset yii2
python main.py --lab 2 --model wide_deep --task 1 --dataset django

# Task 2
python main.py --lab 2 --model mlp --task 2 --dataset yii2
python main.py --lab 2 --model deep_cross --task 2 --dataset tensorflow

# Multi-Task
python main.py --lab 2 --model multitask --task 1 --dataset yii2
```

## 跨项目预测实验

评估模型的泛化能力，训练在一个项目上，测试在另一个项目上。

### 单个跨项目实验

```powershell
# 在 yii2 上训练，在 django 上测试
python models/lab2/cross_project.py --model mlp --task regression --train yii2 --test django

# 分类任务的跨项目预测
python models/lab2/cross_project.py --model wide_deep --task classification --train tensorflow --test yii2
```

### 完整跨项目矩阵

```powershell
# 运行所有可能的跨项目组合
python models/lab2/cross_project.py --model mlp --task regression --full-matrix
python models/lab2/cross_project.py --model deep_cross --task classification --full-matrix
```

结果会保存为 CSV 文件：
- `cross_project_results_regression_mlp.csv`
- `cross_project_results_classification_deep_cross.csv`

## 模型配置

所有模型配置都在 `config.py` 中定义：

```python
LAB2_CONFIG = {
    "training": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "n_epochs": 100,
        "patience": 10,
    },
    
    "task1": {
        "mlp": {
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.2,
        },
        "wide_deep": {
            "deep_dims": [128, 64],
            "dropout_rate": 0.2,
        },
        # ... 其他模型配置
    },
    
    "task2": {
        # 分类任务配置
    },
    
    "multitask": {
        "shared_dims": [128, 64],
        "regression_dims": [32],
        "classification_dims": [32],
        "loss_weights": {
            "regression": 1.0,
            "classification": 1.0,
        }
    }
}
```

### 修改配置

你可以直接编辑 `config.py` 来调整：
- 网络结构（隐藏层大小、层数）
- 训练参数（学习率、批次大小、最大轮数）
- Dropout 比率
- 多任务学习的损失权重

## 输出说明

### Task 1 (回归) 输出

```
=====================================================================
Task 1: MLP for PR Time-to-Close Prediction
=====================================================================
...
Deep Learning Model Evaluation Results (Regression)
============================================================
Mean Absolute Error (MAE): 45.23 hours (1.88 days)
Root Mean Squared Error (RMSE): 67.89 hours (2.83 days)
R² Score: 0.6543 (65.43%)
============================================================
```

### Task 2 (分类) 输出

```
=====================================================================
Task 2: MLP for PR Merge Status Prediction
=====================================================================
...
Task 2: Classification Model Evaluation Results
======================================================================
Accuracy: 0.8234 (82.34%)
Precision (Macro): 0.7891
Recall (Macro): 0.7654
F1 Score (Macro): 0.7765

Confusion Matrix:
[[150  20]
 [ 15 135]]

Detailed Classification Report:
              precision    recall  f1-score   support
  Not Merged       0.91      0.88      0.89       170
      Merged       0.87      0.90      0.89       150
    accuracy                           0.89       320
   macro avg       0.89      0.89      0.89       320
weighted avg       0.89      0.89      0.89       320
======================================================================
```

### Multi-Task 输出

```
Multi-Task Learning Evaluation Results
======================================================================

【Task 1: Regression - TTC Prediction】
  MAE:  43.15 hours (1.80 days)
  RMSE: 65.22 hours (2.72 days)
  R²:   0.6723

【Task 2: Classification - Merge Prediction】
  Accuracy:  0.8345 (83.45%)
  Precision: 0.8012
  Recall:    0.7889
  F1 Score:  0.7945
======================================================================
```

## 模型检查点

训练完成的模型会自动保存到 `checkpoints/` 目录：

```
checkpoints/
├── task1_mlp_yii2.pth
├── task1_wide_deep_django.pth
├── task2_mlp_yii2.pth
├── task2_deep_cross_tensorflow.pth
└── multitask_yii2.pth
```

## 高级用法

### 1. 使用原始数据而非预提取特征

```powershell
python models/lab2/task1.py --model mlp --dataset yii2 --no-extracted
```

### 2. 在不同数据集上训练

```powershell
python models/lab2/task1.py --model wide_deep --dataset django
python models/lab2/task1.py --model wide_deep --dataset tensorflow
python models/lab2/task1.py --model wide_deep --dataset react
```

### 3. 比较多个模型

```powershell
# 训练所有 Task1 模型
python models/lab2/task1.py --model mlp --dataset yii2
python models/lab2/task1.py --model wide_deep --dataset yii2
python models/lab2/task1.py --model deep_cross --dataset yii2
python models/lab2/task1.py --model shared_bottom --dataset yii2
python models/lab2/task1.py --model mmoe --dataset yii2

# 训练所有 Task2 模型
python models/lab2/task2.py --model mlp --dataset yii2
python models/lab2/task2.py --model wide_deep --dataset yii2
python models/lab2/task2.py --model deep_cross --dataset yii2
```

## 实验设计建议

### 1. 模型对比实验
- 在同一数据集上训练所有模型
- 比较性能指标
- 分析各模型的优缺点

### 2. 跨项目泛化实验
- 使用 `cross_project.py` 进行跨项目预测
- 评估模型在不同项目上的泛化能力
- 分析哪些模型泛化性更好

### 3. 多任务学习效果分析
- 比较单任务模型 vs 多任务模型
- 分析共享表示是否有助于提升性能
- 调整损失权重观察影响

### 4. 超参数调优
- 修改 `config.py` 中的超参数
- 比较不同网络结构的效果
- 调整学习率、batch size 等

## 故障排除

### GPU 内存不足

如果遇到 CUDA out of memory 错误：

```python
# 在 config.py 中减小 batch_size
"training": {
    "batch_size": 32,  # 从 64 改为 32
    ...
}
```

### 模型不收敛

1. 降低学习率
2. 增加训练轮数
3. 检查数据预处理
4. 尝试不同的网络结构

### 数据加载错误

确保数据文件存在：
```
data/yii2/PR_extracted_features.xlsx
```

如果没有预提取特征，使用：
```powershell
python models/lab2/task1.py --model mlp --dataset yii2 --no-extracted
```

## 总结

Lab2 提供了完整的深度学习解决方案：
- ✅ 多种先进的神经网络架构
- ✅ Task 1 (回归) 和 Task 2 (分类)
- ✅ 多任务学习
- ✅ 跨项目预测实验
- ✅ 完整的评价指标
- ✅ 灵活的配置系统

通过这些工具，你可以：
1. 训练和评估多种模型
2. 进行跨项目泛化性实验
3. 探索多任务学习的优势
4. 比较不同架构的性能
