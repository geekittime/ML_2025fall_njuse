import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression  # 修改点：导入逻辑回归
from sklearn.preprocessing import StandardScaler  # 修改点：导入标准化工具
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

pr_info_path = os.path.join(path_prefix, 'PR_info.xlsx')
pr_features_path = os.path.join(path_prefix, 'PR_features.xlsx')
author_features_path = os.path.join(path_prefix, 'author_features.xlsx')
pr_info_add_conversation_path = os.path.join(path_prefix, 'PR_info_add_conversation.xlsx')

pr_info = pd.read_excel(pr_info_path)
pr_features = pd.read_excel(pr_features_path)
author_features = pd.read_excel(author_features_path)
pr_info_add_conversation = pd.read_excel(pr_info_add_conversation_path)

# --- 【关键修正】: 修正了author_features的合并逻辑 ---
merged_df = pd.merge(pr_info, pr_features, on='number', how='left')
merged_df = pd.merge(merged_df, author_features, on='number', how='left')
merged_df = pd.merge(merged_df, pr_info_add_conversation, on='number', how='left')


# 处理合并产生的同名列
def handle_duplicate_columns(df):
    for col in df.columns:
        if col.endswith('_y'):
            df.drop(columns=[col], inplace=True)
        elif col.endswith('_x'):
            df.rename(columns={col: col[:-2]}, inplace=True)
    return df


merged_df = handle_duplicate_columns(merged_df)

# 清理无用的索引列
columns_to_drop = [col for col in merged_df.columns if 'Unnamed: 0' in str(col)]
merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print(f"数据合并完成。数据集共有 {merged_df.shape[0]} 行和 {merged_df.shape[1]} 列。")

# --- 2. 数据预处理与特征工程 ---
print("\n步骤 2: 开始数据预处理和特征工程...")

# --- 【关键修正】: 重新启用对state的筛选 ---
merged_df = merged_df[merged_df['state'] == 'closed']
print(f"筛选出 state 为 'closed' 的PR后，剩余 {len(merged_df)} 行。")

# 定义目标变量 y
y = merged_df['merged'].astype(int)

# 定义特征 X
merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])

features = merged_df.select_dtypes(include=np.number).columns.tolist()

# 从特征列表中移除ID和目标变量本身
features_to_remove = ['number', 'author_id', 'project_id', 'merged']
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

# --- 修改点：为逻辑回归增加特征标准化步骤 ---
print("\n步骤 3.5: 为逻辑回归进行特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("特征标准化完成。")

# --- 4. 模型训练 ---
print("\n步骤 4: 开始训练逻辑回归模型...")

# --- 修改点：使用逻辑回归分类器 ---
model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=3000)
model.fit(X_train_scaled, y_train)
print("模型训练完成。")

# --- 5. 预测与评估 ---
print("\n步骤 5: 在测试集上进行预测和评估...")

# 使用标准化后的数据进行预测
predictions = model.predict(X_test_scaled)

# 计算并打印混淆矩阵
print("\n--- 混淆矩阵 ---")
cm = confusion_matrix(y_test, predictions)
print(cm)
print("--------------------")

# 计算总的预测正确率
accuracy = accuracy_score(y_test, predictions)
print(f"\n总预测正确率 (Accuracy): {accuracy:.4f} ({accuracy:.2%})")

# 计算并打印主要的分类指标
print("\n--- 分类报告 ---")
report = classification_report(y_test, predictions, target_names=['Not Merged (0)', 'Merged (1)'])
print(report)
print("--------------------")

# --- 修改点：分析逻辑回归的系数 ---
print("\n--- 逻辑回归模型系数分析 (Top 10 绝对值) ---")
# 将系数与特征名对应
# model.coef_[0] 是因为逻辑回归的系数是一个二维数组
coefficients = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0]
})
# 计算系数的绝对值，用于排序
coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
# 按绝对值降序排序
sorted_coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
print("系数解读：正数表示该特征增大时，PR被合入的概率倾向于增加；负数则相反。")
print(sorted_coefficients.head(10))
print("---------------------------------")