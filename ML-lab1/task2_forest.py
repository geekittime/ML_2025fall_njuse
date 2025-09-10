import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # 修改点：导入随机森林分类器
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report  # 修改点：导入分类评估指标
import warnings
import os

# 忽略一些未来版本的警告，使输出更整洁
warnings.filterwarnings('ignore')

# --- 1. 数据加载与合并 ---
print("步骤 1: 开始加载和合并数据...")

# 使用脚本的绝对路径，使其不受运行位置影响
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
path_prefix = os.path.join(script_dir, 'yii2')

print(f"正在从以下路径加载数据: {path_prefix}")

extracted_path = os.path.join(path_prefix, 'PR_extracted_features.xlsx')
merged_df = pd.read_excel(extracted_path)

print(f"数据合并完成。数据集共有 {merged_df.shape[0]} 行和 {merged_df.shape[1]} 列。")

# --- 2. 数据预处理与特征工程 ---
print("\n步骤 2: 开始数据预处理和特征工程...")

# 2. 定义目标变量 y
# 将布尔列转换为数字0或1
for bool_col in merged_df.select_dtypes(include=bool).columns.tolist():
    merged_df[bool_col] = merged_df[bool_col].astype(int)
y = merged_df['merged']

# 3. 定义特征 X
# 转换时间列，仅用于后续排序
merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])

# 选择所有数值型特征
features = merged_df.select_dtypes(include=np.number).columns.tolist()

# 从特征列表中移除ID和目标变量本身
# 注意：'merged'列本身不能作为特征来预测自己
features_to_remove = ['number', 'merged']
features = [f for f in features if f not in features_to_remove]

X = merged_df[features]

# 将无穷大值(inf)替换为NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# 处理缺失值：使用中位数填充
print("处理缺失值：使用每列的中位数进行填充...")
X.fillna(X.median(), inplace=True)

print(f"最终选定的特征数量: {len(features)}")

# --- 3. 按时间顺序划分数据集 ---
print("\n步骤 3: 按时间顺序划分训练集和测试集...")
merged_df.sort_values('created_at', inplace=True)
X = X.loc[merged_df.index]
y = y.loc[merged_df.index]
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
print(f"训练集大小: {X_train.shape[0]} 行")
print(f"测试集大小: {X_test.shape[0]} 行")

# --- 4. 模型训练 ---
print("\n步骤 4: 开始训练随机森林分类模型...")

# --- 修改点：使用随机森林分类器 ---
# class_weight='balanced' 可以帮助处理数据不平衡问题（如果合入与未合入的PR数量差异很大）
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)
print("模型训练完成。")

# --- 5. 预测与评估 ---
print("\n步骤 5: 在测试集上进行预测和评估...")

# --- 修改点：使用分类评估方法 ---
predictions = model.predict(X_test)

# 计算并打印混淆矩阵
print("\n--- 混淆矩阵 ---")
# [[TN, FP],
#  [FN, TP]]
# TN: 真实为0，预测为0 (正确预测未合入)
# FP: 真实为0，预测为1 (错误预测为合入)
# FN: 真实为1，预测为0 (错误预测为未合入)
# TP: 真实为1，预测为1 (正确预测合入)
cm = confusion_matrix(y_test, predictions)
print(cm)
print("--------------------")

# 计算总的预测正确率
accuracy = accuracy_score(y_test, predictions)
print(f"\n总预测正确率 (Accuracy): {accuracy:.4f} ({accuracy:.2%})")

# 计算并打印主要的分类指标
print("\n--- 分类报告 ---")
# target_names=['Not Merged (0)', 'Merged (1)'] 指定了0和1对应的标签名
report = classification_report(y_test, predictions, target_names=['Not Merged (0)', 'Merged (1)'])
print(report)
print("--------------------")

# --- 特征重要性分析 (与任务一相同) ---
print("\n--- 特征重要性分析 (Top 10) ---")
feature_importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importances.head(10))
print("---------------------------------")