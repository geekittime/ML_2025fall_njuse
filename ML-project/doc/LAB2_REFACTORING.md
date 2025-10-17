# Lab2 重构完成总结

## 主要改进

### 1. ✅ Task1 完善 - 回归任务
**问题**: 之前只有简单的 MLP 模型  
**解决**: 
- ✅ 实现了 5 种神经网络模型：
  - MLP (Multi-Layer Perceptron)
  - Wide & Deep
  - Deep & Cross
  - Shared-Bottom
  - MMoE (Multi-gate Mixture-of-Experts)
- ✅ 统一的评价指标：MAE, MSE, RMSE, R²

### 2. ✅ Task2 新增 - 分类任务
**问题**: 完全缺失Task2的分类任务  
**解决**:
- ✅ 新建 `models/lab2/task2.py`
- ✅ 实现了 3 种分类模型：
  - MLP Classifier
  - Wide & Deep Classifier
  - Deep & Cross Classifier
- ✅ 完整的评价指标：Accuracy, Precision (Macro), Recall (Macro), F1 Score (Macro)

### 3. ✅ 多任务学习独立
**问题**: 多任务模型混在Task1中  
**解决**:
- ✅ 创建独立的 `models/lab2/multitask.py`
- ✅ 实现 Shared-Bottom 多任务架构
- ✅ 同时优化回归和分类两个任务
- ✅ 可配置的损失权重

### 4. ✅ 跨项目预测实验
**问题**: 缺少泛化性实验  
**解决**:
- ✅ 创建 `models/lab2/cross_project.py`
- ✅ 支持单个跨项目实验
- ✅ 支持完整跨项目矩阵
- ✅ 自动生成结果CSV报告

## 新增文件

### 核心代码文件
```
models/lab2/
├── architectures.py       # 所有神经网络架构定义
├── task1.py              # Task1 统一训练脚本 (支持5种模型)
├── task2.py              # Task2 分类任务 (新增)
├── multitask.py          # 多任务学习 (独立出来)
└── cross_project.py      # 跨项目预测实验 (新增)
```

### 文档文件
```
LAB2_GUIDE.md            # Lab2 完整使用指南
LAB2_REFACTORING.md      # 本文档 - 重构总结
```

### 配置更新
```
config.py                # 更新了 LAB2_CONFIG
```

## 架构设计

### models/lab2/architectures.py
集中定义所有神经网络架构：

**回归模型** (5个):
- `MLPRegressor`: 标准MLP
- `WideAndDeepRegressor`: Wide & Deep 架构
- `DeepCrossRegressor`: Deep & Cross 网络
- `SharedBottomRegressor`: Shared-Bottom (可用于多任务)
- `MMoERegressor`: Multi-gate Mixture-of-Experts

**分类模型** (3个):
- `MLPClassifier`
- `WideAndDeepClassifier`
- `DeepCrossClassifier`

**多任务模型** (1个):
- `MultiTaskModel`: 同时处理回归和分类

### 统一的训练流程

所有模型都遵循相同的流程：
1. 加载数据
2. 特征预处理
3. 时序划分
4. 模型构建
5. 训练(带早停)
6. 评估
7. 保存模型

## 使用方法

### 方法 1: 直接运行任务脚本

```powershell
# Task1 - 回归
python models/lab2/task1.py --model mlp --dataset yii2
python models/lab2/task1.py --model wide_deep --dataset django
python models/lab2/task1.py --model deep_cross --dataset tensorflow
python models/lab2/task1.py --model shared_bottom --dataset yii2
python models/lab2/task1.py --model mmoe --dataset yii2

# Task2 - 分类
python models/lab2/task2.py --model mlp --dataset yii2
python models/lab2/task2.py --model wide_deep --dataset django
python models/lab2/task2.py --model deep_cross --dataset tensorflow

# 多任务学习
python models/lab2/multitask.py --dataset yii2

# 跨项目预测
python models/lab2/cross_project.py --model mlp --task regression --train yii2 --test django
python models/lab2/cross_project.py --model wide_deep --task classification --full-matrix
```

### 方法 2: 通过 main.py 统一入口

```powershell
# Task1
python main.py --lab 2 --model mlp --task 1 --dataset yii2
python main.py --lab 2 --model wide_deep --task 1 --dataset django

# Task2
python main.py --lab 2 --model mlp --task 2 --dataset yii2
python main.py --lab 2 --model deep_cross --task 2 --dataset tensorflow

# 多任务
python main.py --lab 2 --model multitask --task 1 --dataset yii2
```

## 配置系统

### config.py 更新

```python
LAB2_CONFIG = {
    "training": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "n_epochs": 100,
        "patience": 10,
        "device": "auto",
    },
    
    "task1": {
        "mlp": {...},
        "wide_deep": {...},
        "deep_cross": {...},
        "shared_bottom": {...},
        "mmoe": {...},
    },
    
    "task2": {
        "mlp": {...},
        "wide_deep": {...},
        "deep_cross": {...},
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

所有超参数都可以在这里统一配置。

## 评价指标

### Task1 (回归)
- **MAE** (Mean Absolute Error): 平均绝对误差
- **MSE** (Mean Squared Error): 均方误差  
- **RMSE** (Root Mean Squared Error): 均方根误差
- **R²** (Coefficient of Determination): 决定系数

### Task2 (分类)
- **Accuracy**: 准确率
- **Precision (Macro)**: 精确度 (宏平均)
- **Recall (Macro)**: 召回率 (宏平均)
- **F1 Score (Macro)**: F1分数 (宏平均)
- **Confusion Matrix**: 混淆矩阵
- **Classification Report**: 详细分类报告

### 多任务学习
同时显示两个任务的所有指标

## 跨项目预测

### 单个实验
在一个项目上训练，另一个项目上测试：

```powershell
python models/lab2/cross_project.py --model mlp --task regression --train yii2 --test django
```

### 完整矩阵
自动运行所有可能的训练-测试组合：

```powershell
python models/lab2/cross_project.py --model mlp --task regression --full-matrix
```

输出 CSV 文件包含所有实验结果，便于分析。

## 实验建议

### 1. 模型对比实验
在同一数据集上训练所有模型，比较性能：

```powershell
# Task1 所有模型
python models/lab2/task1.py --model mlp --dataset yii2
python models/lab2/task1.py --model wide_deep --dataset yii2
python models/lab2/task1.py --model deep_cross --dataset yii2
python models/lab2/task1.py --model shared_bottom --dataset yii2
python models/lab2/task1.py --model mmoe --dataset yii2

# Task2 所有模型
python models/lab2/task2.py --model mlp --dataset yii2
python models/lab2/task2.py --model wide_deep --dataset yii2
python models/lab2/task2.py --model deep_cross --dataset yii2
```

### 2. 跨项目泛化实验
评估模型在不同项目上的泛化能力：

```powershell
python models/lab2/cross_project.py --model mlp --task regression --full-matrix
python models/lab2/cross_project.py --model deep_cross --task classification --full-matrix
```

### 3. 多任务学习分析
比较单任务 vs 多任务：

```powershell
# 单任务
python models/lab2/task1.py --model mlp --dataset yii2
python models/lab2/task2.py --model mlp --dataset yii2

# 多任务
python models/lab2/multitask.py --dataset yii2

# 比较性能差异
```

## 代码质量改进

### 模块化
- ✅ 网络架构独立在 `architectures.py`
- ✅ 训练逻辑在各任务脚本中
- ✅ 基类提供通用功能

### 可扩展性
- ✅ 添加新模型只需在 `architectures.py` 中定义
- ✅ 在任务脚本中注册即可使用
- ✅ 配置统一管理

### 代码复用
- ✅ Task1 和 Task2 共享相同的训练流程
- ✅ 所有模型使用统一的 Lab2BaseModel
- ✅ 数据加载和预处理统一

## 文件对应关系

| 功能                | 原文件                   | 新文件                   | 状态   |
| ------------------- | ------------------------ | ------------------------ | ------ |
| Task1 MLP           | task1/task1_Linear.py    | task1.py (mlp)           | ✅ 重构 |
| Task1 Wide&Deep     | task1/task1_Wide&Deep.py | task1.py (wide_deep)     | ✅ 重构 |
| Task1 DeepCross     | task1/task1_deepcross.py | task1.py (deep_cross)    | ✅ 重构 |
| Task1 Shared-Bottom | -                        | task1.py (shared_bottom) | ✅ 新增 |
| Task1 MMoE          | -                        | task1.py (mmoe)          | ✅ 新增 |
| Task2 MLP           | task2/task2_MLP.py       | task2.py (mlp)           | ✅ 重构 |
| Task2 Wide&Deep     | task2/task2_widedeep.py  | task2.py (wide_deep)     | ✅ 重构 |
| Task2 DeepCross     | task2/task2_deepcross.py | task2.py (deep_cross)    | ✅ 重构 |
| 多任务学习          | Multitask.py             | multitask.py             | ✅ 独立 |
| 跨项目预测          | -                        | cross_project.py         | ✅ 新增 |

## 后续可以做的

### 可选扩展
1. **可视化工具**: 添加训练曲线、特征重要性可视化
2. **更多模型**: Transformer, Attention机制等
3. **超参数搜索**: 自动超参数优化
4. **模型集成**: 集成多个模型提升性能
5. **实时预测API**: 部署模型为Web服务

### 实验分析
1. **消融实验**: 分析各个组件的贡献
2. **特征分析**: 哪些特征最重要
3. **错误分析**: 分析预测错误的案例
4. **时间分析**: 不同时间段的预测性能

## 总结

这次 Lab2 重构完成了以下目标：

### ✅ 已完成
1. Task1 支持 5 种神经网络模型
2. Task2 完整实现（之前缺失）
3. 多任务学习独立模块
4. 跨项目预测实验框架
5. 完整的评价指标
6. 统一的配置管理
7. 清晰的代码结构
8. 详细的使用文档

### 优势
- **完整性**: 覆盖所有需求
- **可扩展**: 易于添加新模型
- **可维护**: 代码结构清晰
- **易用性**: 统一的接口
- **专业性**: 符合工程标准

现在你有一个完整、专业、易用的深度学习实验框架！🎉
