# Lab2 综述

## 概述

Lab2 实现了多种深度学习模型用于 Pull Request 分析：

### 完成情况
Level 1：完成

Level 2：特征工程、跨项目评估

Level 3：多任务网络，完成

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

### 使用特征
默认情况下使用预处理特征，由`utils/feature_extract.py`（用于实验给出的数据集）或`utils/crawled_feature_extract.py`（用于使用`utils/repo_crawler.py`爬取得到的数据）处理得到。

预处理特征继承自Lab1，为手动筛选的一组简单特征，包括PR的修改行数、文件数、最后一条评论时间、PR作者的所有PR被合并的比例等，特征数尽可能少以提高模型的泛用性。

在没有预处理特征的情况下，自动合并所有文件内的内容作为特征。

此外，模型在使用数据之前，还会进行一些简单的处理，如对时间相关的特征进行log1p变换、滤除缺省值等，并按PR创建时间以一定比例（默认4:1）切分训练与测试数据。

## 分工
陈哲敏 231250123：特征工程、部分代码与文档编写

王彬宇 231250166：模型代码编写

何棋 231250083：代码重构工作、部分文档编写

刘柏成 231250098：部分文档编写

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

### 0. 环境
要求Python 3.9+，并使用以下命令安装依赖：

```powershell
pip install -r requirements.txt
```

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
python main.py --lab 2 --model - --task multitask --dataset yii2
```

## 跨项目预测实验

评估模型的泛化能力，训练在一个项目上，测试在另一个项目上。

支持选择特定两个项目进行跨项目测试，也可以批量运行所有的跨项目组合，评估整体效果。

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

## 实验结果

### Task 1 (回归)(MLP模型，yii2)

```
=====================================================================
Task 1: MLP for PR Time-to-Close Prediction
=====================================================================
...
============================================================
Deep Learning Model Evaluation Results (Regression)
============================================================
Mean Absolute Error (MAE): 48.30 hours (2.01 days)
Root Mean Squared Error (RMSE): 121.09 hours (5.05 days)
R² Score: 0.4496 (44.96%)
============================================================
```

### Task 2 (分类)(MLP模型，yii2)

```
=====================================================================
Task 2: MLP for PR Merge Status Prediction
=====================================================================
...
======================================================================
Task 2: Classification Model Evaluation Results
======================================================================
Accuracy: 0.8744 (87.44%)
Precision (Macro): 0.8342
Recall (Macro): 0.7795
F1 Score (Macro): 0.8016

Confusion Matrix:
[[ 214  137]
 [  63 1178]]

Detailed Classification Report:
              precision    recall  f1-score   support

  Not Merged       0.77      0.61      0.68       351
      Merged       0.90      0.95      0.92      1241

    accuracy                           0.87      1592
   macro avg       0.83      0.78      0.80      1592
weighted avg       0.87      0.87      0.87      1592

======================================================================
```

### Multi-Task (yii2)

```
======================================================================
Multi-Task Learning Evaluation Results
======================================================================

【Task 1: Regression - TTC Prediction】
  MAE:  73.53 hours (3.06 days)
  RMSE: 164.01 hours (6.83 days)
  R²:   0.0064

【Task 2: Classification - Merge Prediction】
  Accuracy:  0.8999 (89.99%)
  Precision: 0.9129
  Recall:    0.7447
  F1 Score:  0.7951

  Confusion Matrix:
[[ 131  132]
 [  10 1146]]

  Classification Report:
              precision    recall  f1-score   support

  Not Merged       0.93      0.50      0.65       263
      Merged       0.90      0.99      0.94      1156

    accuracy                           0.90      1419
   macro avg       0.91      0.74      0.80      1419
weighted avg       0.90      0.90      0.89      1419

======================================================================
```

结果显示，多任务网络在任务1上表现不良，但在任务2上表现仍然出色

### Cross Project
一份涉及5个项目的任务1跨项目测试结果已经在项目根目录下。

从结果来看，模型一般有一定的泛化能力，可以在不同的项目下得到可观的表现，但不是对于所有项目都如此。

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
